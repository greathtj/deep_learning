import os
import pickle

import numpy as np
import librosa
import tensorflow as tf

import sounddevice as sd
import soundfile as sf

import os
import numpy as np
from sklearn.model_selection import train_test_split

class audioClassifier():
    def __init__(self, sample_rate=44100, target_duration=2):
        self.sample_rate = sample_rate
        self.target_duration = target_duration

    def loadModel(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_list = self.loadClasses(model_path)

    def loadClasses(self, model_path):
        base_name, old_extension = os.path.splitext(model_path)
        new_extension = '.pkl'
        new_file_path = base_name + new_extension

        # Load the list from the file
        class_list = []
        with open(new_file_path, 'rb') as file:
            class_list = pickle.load(file)

        return class_list

    def preprocess_audio_clip(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        
        # Resize the audio clip to the target duration
        num_samples = int(self.target_duration * self.sample_rate)
        if len(audio) > num_samples:
            audio = audio[:num_samples]
        else:
            padding = num_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Apply spectrogram transformation
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=128)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram

    # Function to perform classification
    def classify_audio_clip(self, file_path):
        spectrogram = self.preprocess_audio_clip(file_path)
        input_data = spectrogram[np.newaxis, ..., np.newaxis]  # Reshape to (1, height, width, channels)
        
        # Perform classification
        predictions = self.model.predict(input_data)
        class_index = np.argmax(predictions)

        if class_index < len(self.class_list):
            return self.class_list[class_index]
        else:
            return "Unknow"


class audioProcess():
    def __init__(self, microphone_index, file_path, sample_rate=44100, target_duration=2):
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.output_file = file_path
        self.microphone_index = microphone_index

    def getInputDeviceID(self):
        # Get a list of available input devices (microphones)
        input_devices = sd.query_devices()

        print("Available devices:")
        for i, device in enumerate(input_devices):
            print(f"{i}. {device['name']}")
        
        input_index = int(input('Select one of the indices: '))
        self.microphone_index = input_index
        print(f'Selected device index is {input_index}')

        return input_index
    
    def getInputDeviceList(self):
        # Get a list of available input devices (microphones)
        input_devices = sd.query_devices()
        device_list = []
        for i, device in enumerate(input_devices):
            device_str = f'{i}: {device["name"]}'
            device_list.append(device_str)

        return device_list
    
    def splitAudioFile(self, input_file, output_folder, segment_duration=2):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        data, sample_rate = sf.read(input_file)
        
        num_segments = len(data) // (sample_rate * segment_duration)
        
        for i in range(num_segments):
            start_sample = i * sample_rate * segment_duration
            end_sample = (i + 1) * sample_rate * segment_duration
            segment = data[start_sample:end_sample]
            
            output_file = os.path.join(output_folder, f'segment_{i+1}.wav')
            sf.write(output_file, segment, sample_rate)

    def captureAudio(self):
        print("Recording audio...")

        # Record audio from the specified microphone
        audio_data = sd.rec(int(self.target_duration * self.sample_rate), 
                            samplerate=self.sample_rate, 
                            channels=1, 
                            dtype='float32', 
                            device=self.microphone_index)
        sd.wait()  # Wait until recording is finished

        print("Recording finished.")

        # Save the recorded audio as a .wav file using soundfile
        sf.write(self.output_file, audio_data, self.sample_rate)

        print(f"Audio saved as {self.output_file}")

    def captureAudioByIndex(self, inputIndex, outputFile, samplingRate=44100, targetDuration=2, inputLevel=1):
        print("Recording audio...")

        # Record audio from the specified microphone
        audio_data = sd.rec(int(targetDuration * samplingRate), 
                            samplerate=samplingRate, 
                            channels=1, 
                            dtype='float32', 
                            device=inputIndex)
        sd.wait()  # Wait until recording is finished
        audio_data *= inputLevel

        print("Recording finished.")

        # Save the recorded audio as a .wav file using soundfile
        sf.write(outputFile, audio_data, samplingRate)

        print(f"Audio saved as {outputFile}")

        return audio_data

    def adjust_input_level(self):
        # Set input level (gain)
        input_level = float(input("Enter the desired input level (0 to 2): "))
        if input_level < 0 or input_level > 2:
            print("Invalid input level. Using default value of 1.")
            input_level = 1

        return input_level

    def record_and_save_audio(self, device_name, input_level, output_file):
        # Set the parameters for audio capture
        SAMPLE_RATE = 44100
        DURATION = 2  # Duration of the recording in seconds

        print("Recording audio...")

        # Record audio from the microphone
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=device_name)
        # audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=device_name, gain=input_level)
        sd.wait()  # Wait until recording is finished

        print("Recording finished.")

        # # Adjust input level (gain)
        # input_level = float(input("Enter the desired input level adjustment (e.g., 0.5 for half, 2.0 for double): "))
        audio_data *= input_level

        # Save the recorded audio as a .wav file
        sf.write(output_file, audio_data, SAMPLE_RATE)

        print(f"Audio saved as {output_file}")



class classifierTrainer():
    def __init__(self, sample_rate=44100, audio_duration=2):
        self.sample_rate = sample_rate
        self.target_duration = audio_duration

    def buildModel(self, data_root, model_path, epochs, batches):
        self.class_list = self.getClasses(data_root)
        print(f'>> The classes in your root data folder = {self.class_list}.')
        self.data = self.prepareData(data_root)
        print(f'>> Your data is organized as:\n{self.data}')
        X_train, X_test, y_train, y_test = self.transformTrainingData()
        model = self.trainModel(X_train, X_test, y_train, y_test, epochs=epochs, batches=batches, valid_split=0.2)
        self.saveTrainedModel(model, model_path)
        self.saveTrainedClasses(model_path)

    def saveTrainedClasses(self, model_path):
        base_name, old_extension = os.path.splitext(model_path)
        new_extension = '.pkl'
        new_file_path = base_name + new_extension
        with open(new_file_path, 'wb') as file:
            pickle.dump(self.class_list, file)
        
        print(f'The classes are saved in {new_file_path}')

    def saveTrainedModel(self, model, save_path):
        # Save the model
        model.save(save_path)
        print(f'The trained model is saved as {save_path}')

    def trainModel(self, X_train, X_test, y_train, y_test, epochs=10, batches=32, valid_split=0.2):
        num_classes = len(self.class_list)
        # Build the model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2], 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, batch_size=batches, epochs=epochs, validation_split=valid_split)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Test accuracy: {test_accuracy} with Test loss: {test_loss}')

        return model

    def transformTrainingData(self):
        # Preprocess and load data
        X = []
        y = []
        for file_path, label in self.data:
            spectrogram = self.preprocess_audio_clip(file_path)
            X.append(spectrogram)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Reshape the data to 4D tensor (batch_size, height, width, channels)
        X = X[..., np.newaxis]

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def prepareData(self, data_root):
        # Load data
        data = []
        for label, class_name in enumerate(self.class_list):
            class_dir = os.path.join(data_root, class_name)  # Replace with the actual path
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                data.append((file_path, label))

        return data

    def getClasses(self, data_root):
        root_folder = data_root
        subfolder_names = [name for name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, name))]
        
        return subfolder_names

    # Function to load and preprocess audio clip
    def preprocess_audio_clip(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        
        # Resize the audio clip to the target duration
        num_samples = int(self.target_duration * self.sample_rate)
        if len(audio) > num_samples:
            audio = audio[:num_samples]
        else:
            padding = num_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Apply spectrogram transformation
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=128)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram
