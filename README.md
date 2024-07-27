# Speech emotion recognition with deep convolutional neural networks
Complete code for a Python Speech Emotion Recognition model using CNN's, tested with RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song),
CaFE (Canadian French Emotional) and MESD (Mexican Emotional Speech Database) datasets.  
Current CNN model based on the [article](https://www.sciencedirect.com/science/article/abs/pii/S1746809420300501) from D Issa, MF Demirci and A Yazici.  
Todo: Increasing the current precision for RAVDESS dataset to 80-90%, if new literature has upgrades to accuracy using CNN's.  
Todo: Increasing the current precision for CAFE dataset.  
Links to the datasets: [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio), [CaFE](https://zenodo.org/records/1478765), [MESD](https://www.kaggle.com/datasets/saurabhshahane/mexican-emotional-speech-database-mesd)  
The speech_recognition.py file consists of three main parts:  
1. Feature extraction (using dataframe_generators.py file)
2. Creating the CNN model
3. Training the model and analysis of the results

Current accuracies for RAVDESS, CaFE and MESD (respectively): 71%, 63%, 84%
