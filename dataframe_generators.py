import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def feature_extraction(path_dict, counter, offs=0.5):
    x, sr = librosa.load(path_dict[counter],duration=2.5, res_type = 'kaiser_fast', sr=22050*2, offset=offs)
    mel_spectrogram_feature = librosa.feature.melspectrogram(y=x, sr=sr) 
    mel_spectrogram_feature = librosa.power_to_db(mel_spectrogram_feature)
    mel_spectrogram_feature = np.mean(mel_spectrogram_feature.T, axis = 0)
    mfcc_feature = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    mfcc_feature = np.mean(mfcc_feature.T, axis = 0)
    chromagram_feature = librosa.feature.chroma_stft(y=x, sr=sr) 
    chromagram_feature = np.mean(chromagram_feature.T, axis = 0)
    spectral_contrast_feature = librosa.feature.spectral_contrast(y=x, sr=sr) 
    spectral_contrast_feature = np.mean(spectral_contrast_feature.T, axis = 0)
    tonnetz_feature = librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sr) 
    tonnetz_feature = np.mean(tonnetz_feature.T, axis = 0)
    return [mfcc_feature, chromagram_feature, mel_spectrogram_feature, spectral_contrast_feature, tonnetz_feature]

def feature_normalization(feature_array):
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(0, len(feature_array)):
        feature_array[i] = scaler.fit_transform(feature_array[i].reshape(-1,1))
    return feature_array

def create_feature_dataframe_ravdess(dataframe_filename):
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

def create_feature_dataframe_cafe(dataframe_filename):
    #Create audio dataframe
    emotion = []
    gender = []
    actor = []
    file_path = []
    for i in range(1,8):
        if i != 4:
            filename = os.listdir("CaFE_48k/"+str(i)+"/Fort") #Emociones en Fuerte 
        else:
            filename = os.listdir("CaFE_48k/"+str(i)) #Emociones en Fuerte 
        for f in filename: #By each actor file
            part = f.split('.')[0].split('-')
            emotion.append(str(part[1]))
            actor.append(int(part[0]))
            bg = int(part[0])
            if bg%2 == 0:
                bg = "female"
            else:
                bg = "male"
            gender.append(bg)
            if i != 4:
                file_path.append("CaFE_48k/"+str(i)+"/Fort/"+ f)
            else:
                file_path.append("CaFE_48k/"+str(i)+"/"+ f)
    audio_df = pd.DataFrame(emotion)
    audio_df = audio_df.replace({"C":'anger', "D":'disgust', "J":'happiness', "N":'neutral', "P":'fear', "S":'surprise', "T":'sadness'})
    audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
    audio_df.columns = ['gender','emotion','actor']
    audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)
    ipd.display(audio_df)
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

def create_feature_dataframe_mesd(dataframe_filename):
    emotion = []
    gender = []
    actor = []
    file_path = []
    filename = os.listdir("MESD/MESD_Files")
    for f in filename: #By each file
        part = f.split('.')[0].split('_')
        print(part)
        emotion.append(str(part[0]))
        actor.append(part[1])
        bg = part[1]
        if bg == "F":
            bg = "female"
        elif bg == "M":
            bg = "male"
        else:
            bg = "child"
        gender.append(bg)
        file_path.append("MESD/MESD_Files/"+ f)
    audio_df = pd.DataFrame(emotion)
    audio_df = audio_df.replace({"Anger":'anger', "Disgust":'disgust', "Fear":'fear', "Happiness":'hapiness', "Neutral":'neutral', "Sadness":'sadness'})
    audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
    audio_df.columns = ['gender','emotion','actor']
    audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)
    ipd.display(audio_df)
    #Add features to dataframe
    #Uncomment code if you don't have audio_df_full.csv file
    df = pd.DataFrame(columns=['MFCC', 'Chromagram', 'Mel_spectrogram', 'Spectral_contrast', 'Tonnetz'])
    counter = 0
    path_dict = {}
    rows = []
    for index, path in enumerate(audio_df.path):
        path_dict[counter] = path
        counter = counter + 1
    results = Parallel(n_jobs=4)(delayed(feature_extraction)(path_dict, counter, 0) for counter in range(0, len(path_dict)))
    for i in range(0, len(results)):
        df.loc[i] = results[i]
    print(len(df))
    df.head()
    df_combined = pd.concat([audio_df,df],axis=1)
    ipd.display(df_combined)
    df_combined.to_csv(dataframe_filename, index=False)