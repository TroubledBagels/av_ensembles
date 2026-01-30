import torch
import torch.nn as nn
from datasets.AVE_DS import AVEAudio, make_mono, DSType, AVEVideo
from models.AVESquare import AudioClassifier, VideoClassifier, AVESquare
import pathlib
import os
import tqdm
from torchvision import transforms
import torchaudio
import argparse

a_transform = transforms.Compose([
    torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000),
    make_mono,
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512),
    torchaudio.transforms.AmplitudeToDB()
])

v_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

home = str(pathlib.Path.home())
data_dir = os.path.join(home, 'data', 'AVE_Dataset')
tr_ds_a = AVEAudio(ds_dir=data_dir, transform=a_transform, dstype=DSType.TRAIN_VAL)  # type=2 for train/val set
tr_ds_v = AVEVideo(ds_dir=data_dir, dstype=DSType.TRAIN_VAL, transform=v_transform)  # type=2 for train/val set
te_ds_a = AVEAudio(ds_dir=data_dir, transform=a_transform, dstype=DSType.TEST)  # type=4 for test set
te_ds_v = AVEVideo(ds_dir=data_dir, dstype=DSType.TEST, transform=v_transform)  # type=4 for test set
print(f"Dataset size: {len(tr_ds_a)}")

model = AVESquare()
print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train_classifiers(tr_ds_a, tr_ds_v, te_ds_a, te_ds_v, 20, 1e-3, device)