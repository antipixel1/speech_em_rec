import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
from IPython.display import Audio
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import soundfile as sf 
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import sklearn.model_selection as model_selection
import sklearn.tree as tree
from joblib import Parallel, delayed
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Activation
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.regularizers import l2
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import re
from ast import literal_eval
import visualkeras

def feature_extraction(path_dict, counter):
    x, sr = librosa.load(path_dict[counter],duration=2.5, res_type = 'kaiser_fast', sr=22050*2, offset=0.5)
    mel_spectrogram_feature = librosa.feature.melspectrogram(y=x, sr=sr) 
    mel_spectrogram_feature = librosa.power_to_db(mel_spectrogram_feature)
    #temporally average spectrogram
    mel_spectrogram_feature = np.mean(mel_spectrogram_feature.T, axis = 0)
    mfcc_feature = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    #mfcc_feature = librosa.power_to_db(mfcc_feature)
    mfcc_feature = np.mean(mfcc_feature.T, axis = 0)
    #stft =  np.abs(librosa.stft(x))
    chromagram_feature = librosa.feature.chroma_stft(y=x, sr=sr) 
    #chromagram_feature = librosa.power_to_db(chromagram_feature) 
    chromagram_feature = np.mean(chromagram_feature.T, axis = 0)
    spectral_contrast_feature = librosa.feature.spectral_contrast(y=x, sr=sr) 
    #spectral_contrast_feature = librosa.power_to_db(spectral_contrast_feature)
    spectral_contrast_feature = np.mean(spectral_contrast_feature.T, axis = 0)
    tonnetz_feature = librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sr) 
    #tonnetz_feature = librosa.power_to_db(tonnetz_feature)
    tonnetz_feature = np.mean(tonnetz_feature.T, axis = 0)
    return [mfcc_feature, chromagram_feature, mel_spectrogram_feature, spectral_contrast_feature, tonnetz_feature]

def feature_normalization(feature_array):
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(0, len(feature_array)):
        feature_array[i] = scaler.fit_transform(feature_array[i].reshape(-1,1))
    return feature_array

def create_feature_dataframe(dataframe_filename):
    #Iterate over all the files in the dataset and load the data
    emotion = []
    gender = []
    actor = []
    file_path = []
    for i in range(1,25):
        filename = os.listdir("actor_folders/Actor_"+str(i))
        for f in filename: #By each actor file
            part = f.split('.')[0].split('-')
            emotion.append(int(part[2]))
            actor.append(int(part[6]))
            bg = int(part[6])
            if bg%2 == 0:
                bg = "female"
            else:
                bg = "male"
            gender.append(bg)
            file_path.append("actor_folders/Actor_" + str(i) + '/' + f)
    audio_df = pd.DataFrame(emotion)
    audio_df = audio_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
    audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
    audio_df.columns = ['gender','emotion','actor']
    audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)
    ipd.display(audio_df)
    #Now, extract the features
    df = pd.DataFrame(columns=['MFCC', 'Chromagram', 'Mel_spectrogram', 'Spectral_contrast', 'Tonnetz'])
    counter = 0
    path_dict = {}
    rows = []
    for index, path in enumerate(audio_df.path):
        path_dict[counter] = path
        counter = counter + 1
    results = Parallel(n_jobs=4)(delayed(feature_extraction)(path_dict, counter) for counter in range(0, len(path_dict)))
    for i in range(0, len(results)):
        df.loc[i] = results[i]
    print(len(df))
    df.head()
    df_combined = pd.concat([audio_df,df],axis=1)
    ipd.display(df_combined)
    df_combined.to_csv(dataframe_filename, index=False)

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

def create_train_test_split(df_combined, lb):
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
    ipd.display(X_train)
    ipd.display(y_train)
    ipd.display(X_test)
    ipd.display(y_test)
    for column in X_train:
        X_train[column] = clean_column_data(X_train[column])
    for column in X_test:
        X_test[column] = clean_column_data(X_test[column])
    X_train = X_train.applymap(lambda x: np.fromstring(x, dtype=float, sep=','))
    X_test = X_test.applymap(lambda x: np.fromstring(x, dtype=float, sep=','))
    print(len(X_train['Mel_spectrogram'].iloc[0]))
    print(X_train['Mel_spectrogram'].iloc[0])
    print(len(X_train['Mel_spectrogram'].iloc[0]))
    print(type(X_train['Mel_spectrogram'].iloc[0]))
    print("MFCC min and max values: ", np.min(X_train['MFCC'].iloc[0]), np.max(X_train['MFCC'].iloc[0]))
    print("Chromagram min and max values: ", np.min(X_train['Chromagram'].iloc[0]), np.max(X_train['Chromagram'].iloc[0]))
    print("Mel_spectrogram min and max values: ", np.min(X_train['Mel_spectrogram'].iloc[0]), np.max(X_train['Mel_spectrogram'].iloc[0]))
    print("Spectral_contrast min and max values: ", np.min(X_train['Spectral_contrast'].iloc[0]), np.max(X_train['Spectral_contrast'].iloc[0]))
    print("Tonnetz min and max values: ", np.min(X_train['Tonnetz'].iloc[0]), np.max(X_train['Tonnetz'].iloc[0]))
    #plt.hist(X_train['MFCC'].iloc[0], bins=20)
    #plt.show()
    #plt.hist(X_train['Spectral_contrast'].iloc[0], bins=20)
    #plt.show()
    X_train = pd.concat([pd.DataFrame(X_train['MFCC'].tolist()), pd.DataFrame(X_train['Chromagram'].tolist()), pd.DataFrame(X_train['Mel_spectrogram'].tolist()), pd.DataFrame(X_train['Spectral_contrast'].tolist()), pd.DataFrame(X_train['Tonnetz'].tolist())], axis=1)
    X_test = pd.concat([pd.DataFrame(X_test['MFCC'].tolist()), pd.DataFrame(X_test['Chromagram'].tolist()), pd.DataFrame(X_test['Mel_spectrogram'].tolist()), pd.DataFrame(X_test['Spectral_contrast'].tolist()), pd.DataFrame(X_test['Tonnetz'].tolist())], axis=1)
    # # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # CNN REQUIRES INPUT AND OUTPUT ARE NUMBERS
    y_train = to_categorical(lb.fit_transform(y_train.ravel()), num_classes=8)
    y_test = to_categorical(lb.fit_transform(y_test.ravel()), num_classes=8)
    X_train = X_train[:,:,np.newaxis]
    X_test = X_test[:,:,np.newaxis]
    return [X_train, X_test, y_train, y_test]

def main():
    print("Speech recognition test")
    #Add features to dataframe
    #Uncomment next line if you don't have audio_df_full_upgraded.csv file
    #create_feature_dataframe('audio_df_full_upgraded.csv')
    df_combined = pd.read_csv('audio_df_full_upgraded.csv')
    lb = LabelEncoder()
    X_train, X_test, y_train, y_test = create_train_test_split(df_combined, lb)
    #Initial model
    # CNN topology - adapted to our paper's model
    # BUILD 1D CNN LAYERS
    model = Sequential()
    model.add(Conv1D(256, kernel_size=(5), input_shape=(X_train.shape[1],1), strides=1, padding='same'))
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
    model.add(Dense(8))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    opt = keras.optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    #visualkeras.layered_view(model, scale_xy=1, scale_z=1, max_z=100, max_xy=100, type_ignore=[Dropout, Activation, BatchNormalization], legend=True, to_file='outputCNN2.png')

    # TRAINING
    # Set callback functions to early stop training and save the best model so far
    callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    # Train the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=700, validation_data=(X_test, y_test), callbacks=callbacks)
    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # EVALUATION
    # Load best model
    model.load_weights('best_model.h5')
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test acc:", results)
    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(X_test[:3])
    print("predictions shape:", predictions.shape)
    # PREDICTION
    # Load best model
    model.load_weights('best_model.h5')
    # Predict
    y_pred = model.predict(X_test)
    # Convert predictions classes to one hot vectors
    y_pred_classes = np.argmax(y_pred,axis = 1)
    # Convert validation observations to one hot vectors
    y_true = np.argmax(y_test,axis = 1)
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_true, y_pred_classes, normalize='true')
    # plot the confusion matrix
    ConfusionMatrixDisplay(confusion_mtx, display_labels=lb.classes_).plot()
    plt.show()
    # Classification report
    print(classification_report(y_true, y_pred_classes))
    
if __name__ == "__main__":
    main()