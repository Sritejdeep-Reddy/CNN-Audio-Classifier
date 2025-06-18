# You are free to either implement both test() and evaluate() function, or implement test_batch() and evaluate_batch() function. Apart from the 2 functions which you must mandatorily implement, you are free to implement some helper functions as per your convenience.

# Import all necessary python libraries here
# Do not write import statements anywhere else
# !pip install librosa
import librosa
import os
import pandas as pd

import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

TEST_DATA_DIRECTORY_ABSOLUTE_PATH = "/Users/rajabhimanyu/Desktop/audio_dataset/val/siren"
OUTPUT_CSV_ABSOLUTE_PATH = "/Users/rajabhimanyu/Desktop/assasas.csv"
# The above two variables will be changed during testing. The current values are an example of what their contents would look like.

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, padding=0, stride = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding =0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding =1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding =0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding =0, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding =0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.res1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, bias = False, stride=1)
        self.fc1 = nn.Linear(in_features=2240, out_features=640)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=640, out_features=320)
        self.fc3 = nn.Linear(in_features=320, out_features=160)
        self.fc4 = nn.Linear(in_features=160, out_features=13)
    def forward(self, x):
        y = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
    

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None, save_path=None):
    fig, axs = plt.subplots(1, 1)
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    axs.axis('off')

    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    # image_array= image_array[60:-55 ,80:-65,0:3]

    return image_array


def evaluate(file_path):
    # Write your code to predict class for a single audio file instance here
    target_len = 176400
    waveform, sample_rate = torchaudio.load(file_path)
            # waveform = waveform.to(device)

            #combining all channels into one
    channels = []
    for k in range(waveform.size(0)):
        channels.append(waveform[k])

    final_waveform = torch.stack(channels).mean(dim=0)

    #Time-Stretching
    stretched = librosa.effects.time_stretch(final_waveform.numpy(), rate=final_waveform.shape[0]/target_len)
    stretched = stretched[:target_len:]
    stretched = torch.tensor(stretched)

    #Gammatone
    n_fft = 2048
    win_length = None
    hop_length = 1024
    n_mels = 128

    mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk"
    );

    x = plot_spectrogram(mel_spectrogram(stretched))
    x = torch.tensor(x)
    x = x.float()
    x = x.unsqueeze(0)
    x = x.permute(0,3,2,1)
    # print(x.shape)
    model = torch.load("model_CNN1.pt",map_location=torch.device('cpu'))
    outputs = model(x)
    _, predicted = torch.max(outputs.data, 1)
    v = [4,5,6,7,8,9,10,12,13,1,2,3,11]
    predicted = v[predicted.item()]
    print(predicted)
    return predicted


def evaluate_batch(file_path_batch, batch_size=32):
    # Write your code to predict class for a batch of audio file instances here
    return predicted_class_batch


def test():
    filenames = []
    predictions = []
    # for file_path in os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
    for file_name in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        # prediction = evaluate(file_path)
        absolute_file_name = os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_name)
        prediction = evaluate(absolute_file_name)

        filenames.append(absolute_file_name)
        predictions.append(prediction)
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)


def test_batch(batch_size=32):
    filenames = []
    predictions = []

    # paths = os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH)
    paths = os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH)
    paths = [os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, i) for i in paths]
    
    # Iterate over the batches
    # For each batch, execute evaluate_batch function & append the filenames for that batch in the filenames list and the corresponding predictions in the predictions list.

    # The list "paths" contains the absolute file path for all the audio files in the test directory. Now you may iterate over this list in batches as per your choice, and populate the filenames and predictions lists as we have demonstrated in the test() function. Please note that if you use the test_batch function, the end filenames and predictions list obtained must match the output that would be obtained using test() function, as we will evaluating in that way only.
    
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)


# Uncomment exactly one of the two lines below, i.e. either execute test() or test_batch()
# test()
# test_batch()
