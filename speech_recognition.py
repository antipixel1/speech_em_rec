import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import sklearn.model_selection as model_selection
from joblib import Parallel, delayed
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Activation
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import KFold
import dataframe_generators as df_gens
#import visualkeras #if you want the visual representation of CNN

#Helper method to fix problem with dirty values
def clean_column_data(column_data): 
    column_data = column_data.replace('\n','', regex=True)
    column_data = column_data.replace('\[','', regex=True)
    column_data = column_data.replace('\s*\]','', regex=True)
    column_data = column_data.replace('\s+',',', regex=True)
    column_data = column_data.replace('\,+',',', regex=True)
    column_data = column_data.replace('\,+$','', regex=True)
    column_data = column_data.replace('^\,','', regex=True)
    return column_data

def print_feature_ranges(X_train):
    print(len(X_train['Mel_spectrogram'].iloc[0]))
    print(X_train['Mel_spectrogram'].iloc[0])
    print(len(X_train['Mel_spectrogram'].iloc[0]))
    print(type(X_train['Mel_spectrogram'].iloc[0]))
    print("MFCC min and max values: ", np.min(X_train['MFCC'].iloc[0]), np.max(X_train['MFCC'].iloc[0]))
    print("Chromagram min and max values: ", np.min(X_train['Chromagram'].iloc[0]), np.max(X_train['Chromagram'].iloc[0]))
    print("Mel_spectrogram min and max values: ", np.min(X_train['Mel_spectrogram'].iloc[0]), np.max(X_train['Mel_spectrogram'].iloc[0]))
    print("Spectral_contrast min and max values: ", np.min(X_train['Spectral_contrast'].iloc[0]), np.max(X_train['Spectral_contrast'].iloc[0]))
    print("Tonnetz min and max values: ", np.min(X_train['Tonnetz'].iloc[0]), np.max(X_train['Tonnetz'].iloc[0]))
    
def create_train_test_split(df_combined, lb, number_emotions):
    #Split into train and test
    ipd.display(df_combined)
    train,test = model_selection.train_test_split(df_combined, test_size=0.2, random_state=0,
    stratify=df_combined[['emotion','gender','actor']])
    print("Train shape: ", train.shape)
    print("Test shape: ", test.shape)
    X_train = train.iloc[:,3:]
    y_train = train.iloc[:,:2].drop(columns=['gender'])
    X_test = test.iloc[:,3:]
    y_test = test.iloc[:,:2].drop(columns=['gender'])
    X_train = X_train.drop('path', axis=1)
    X_test = X_test.drop('path', axis=1)
    for column in X_train:
        X_train[column] = clean_column_data(X_train[column])
    for column in X_test:
        X_test[column] = clean_column_data(X_test[column])
    X_train = X_train.applymap(lambda x: np.fromstring(x, dtype=float, sep=','))
    X_test = X_test.applymap(lambda x: np.fromstring(x, dtype=float, sep=','))
    print_feature_ranges(X_train)
    X_train = pd.concat([pd.DataFrame(X_train['MFCC'].tolist()), pd.DataFrame(X_train['Chromagram'].tolist()), pd.DataFrame(X_train['Mel_spectrogram'].tolist()), pd.DataFrame(X_train['Spectral_contrast'].tolist()), pd.DataFrame(X_train['Tonnetz'].tolist())], axis=1)
    X_test = pd.concat([pd.DataFrame(X_test['MFCC'].tolist()), pd.DataFrame(X_test['Chromagram'].tolist()), pd.DataFrame(X_test['Mel_spectrogram'].tolist()), pd.DataFrame(X_test['Spectral_contrast'].tolist()), pd.DataFrame(X_test['Tonnetz'].tolist())], axis=1)
    # # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # Make CNN inputs and outputs numerical
    y_train = to_categorical(lb.fit_transform(y_train.ravel()), num_classes=number_emotions)
    y_test = to_categorical(lb.fit_transform(y_test.ravel()), num_classes=number_emotions)
    X_train = X_train[:,:,np.newaxis]
    X_test = X_test[:,:,np.newaxis]
    return [X_train, X_test, y_train, y_test]

def create_model(input_shape_model, num_classes):
    model = Sequential()
    model.add(Conv1D(256, kernel_size=(5), input_shape=input_shape_model, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(128, kernel_size=(5), strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, kernel_size=(5), strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, kernel_size=(5), strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, kernel_size=(5), strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, kernel_size=(5), strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    opt = keras.optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

def plot_confusion_matrix(predicted_labels_list, y_test_list, lbenc):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list, normalize='true')
    cnf_matrix_scaled = cnf_matrix*100
    ConfusionMatrixDisplay(cnf_matrix_scaled, display_labels=lbenc.classes_).plot()
    plt.show()

def main():
    print("Speech emotion recognition")
    #Add features to dataframe
    #Uncomment next line(s) if you don't have the corresponding audio_df_full_dataset.csv files
    #df_gens.create_feature_dataframe_ravdess('audio_df_full_ravdess.csv')
    #df_gens.create_feature_dataframe_cafe('audio_df_full_cafe.csv')
    df_gens.create_feature_dataframe_mesd('audio_df_full_mesd.csv')
    #Change filename to the output filename of the corresponding feature dataframe function
    df_combined = pd.read_csv('audio_df_full_mesd.csv')
    #Hyperparameters
    num_folds = 5
    loss_function = 'categorical_crossentropy'
    opt = keras.optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)
    batch_s = 32
    n_epochs = 700
    verbosity = 2
    number_emotions = 6 #Very important, must match the number of emotions of the dataset!
    lb = LabelEncoder()
    X_train, X_test, y_train, y_test = create_train_test_split(df_combined, lb, number_emotions)
    input_shape_model = (X_train.shape[1],1)
    acc_folds = []
    loss_folds = []
    inputs = np.concatenate((X_train, X_test), axis=0)
    targets = np.concatenate((y_train, y_test), axis=0)
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_number = 1
    predicted_targets = np.array([])
    actual_targets = np.array([])
    print("Inputs shape: ", np.shape(inputs), "Outputs len: ", np.shape(targets))
    for train, test in kfold.split(inputs, targets):
        #Initial model
        # CNN topology - adapted to CNN model from the paper
        # Build 1D CNN layers
        model = create_model(input_shape_model, number_emotions)
        model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
        #If you need a visual representation of the CNN
        #visualkeras.layered_view(model, scale_xy=1, scale_z=1, max_z=100, max_xy=100, type_ignore=[Dropout, Activation, BatchNormalization], legend=True, to_file='outputCNN2.png')
        # Print fold
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_number} ...')
        # Train the model
        history = model.fit(inputs[train], targets[train], batch_size=batch_s, epochs=n_epochs, verbose=verbosity)
        # Generate generalization metrics
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        print(f'Score for fold {fold_number}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_folds.append(scores[1] * 100)
        loss_folds.append(scores[0])
        # Predict values
        predicted_x = model.predict(inputs[test])
        classes_x = np.argmax(predicted_x, axis=1)
        predicted_targets = np.append(predicted_targets, classes_x)
        targets_true = np.argmax(targets[test], axis=1)
        actual_targets = np.append(actual_targets, targets_true)
        # Increase fold number
        fold_number = fold_number + 1
    plot_confusion_matrix(predicted_targets, actual_targets, lb)
    # Provide average scores
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_folds)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_folds[i]} - Accuracy: {acc_folds[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_folds)} (+- {np.std(acc_folds)})')
    print(f'> Loss: {np.mean(loss_folds)}')
    print('------------------------------------------------------------------------')

if __name__ == "__main__":
    main()