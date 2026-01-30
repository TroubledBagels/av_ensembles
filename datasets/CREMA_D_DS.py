import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision
import torchaudio
import pathlib
import time
from torchvision import transforms

'''
CREMA_D Dataset Loader
Labels come as tuples: (emotion, intensity)
Emotion is encoded as follows:
    NEU: Neutral - 0
    HAP: Happy - 1
    SAD: Sad - 2
    ANG: Anger - 3
    FEA: Fear - 4
    DIS: Disgust - 5

Intensity is encoded as follows:
    LO: Low - 0
    MD: Medium - 1
    HI: High - 2
    XX: Unspecified - -1
'''

def emotion_to_label(emotion_str):
    emotion_dict = {
        'NEU': 0,
        'HAP': 1,
        'SAD': 2,
        'ANG': 3,
        'FEA': 4,
        'DIS': 5
    }
    return emotion_dict.get(emotion_str, -1)

def intensity_to_label(intensity_str):
    intensity_dict = {
        'LO': 0,
        'MD': 1,
        'HI': 2,
        'XX': -1
    }
    return intensity_dict.get(intensity_str, -1)


class CREMADVideo(Dataset):
    def __init__(self, video_dir, transform=None):
        """
        A dataset for loading video files from a directory.

        Args:
            video_dir (str): Path to the directory containing video files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.video_dir = video_dir
        self.transform = transform
        self.video_files = [f for f in sorted(os.listdir(video_dir)) if f.endswith('.mp4')]
        self.labels = []
        for vf in self.video_files:
            parts = vf.split('_')
            emotion = emotion_to_label(parts[2])
            intensity = intensity_to_label(parts[3].split('.')[0])
            self.labels.append((emotion, intensity))

    def __len__(self):
        """
        Returns the total number of video files in the dataset.
        """
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        Retrieves the video file at the specified index.

        Args:
            idx (int): The index of the video file to retrieve.

        Returns:
            The video data as a tensor.
        """
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        video_data, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')

        T = video_data.shape[0]
        idx_t = torch.linspace(0, T-1, steps=32).long()
        video_data = video_data[idx_t]

        video_data = video_data.permute(0, 3, 1, 2).float() / 255.0
        video_data = torch.nn.functional.interpolate(video_data, size=(120, 160), mode='bilinear', align_corners=False)

        if self.transform:
            video_data = self.transform(video_data)

        return video_data, self.labels[idx]

    def trim_classes(self, c_1, c_2):
        """
        Trims the dataset to only include samples with the specified classes.

        Args:
            c_1 (int): The first class to keep.
            c_2 (int): The second class to keep.
        """
        filtered_indices = [i for i, label in enumerate(self.labels) if label[0] == c_1 or label[0] == c_2]
        self.video_files = [self.video_files[i] for i in filtered_indices]
        self.labels = [self.labels[i] for i in filtered_indices]

class CREMADAudio(Dataset):
    def __init__(self, audio_dir, transform=None):
        """
        A dataset for loading audio files from a directory.

        Args:
            audio_dir (str): Path to the directory containing audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.audio_dir = audio_dir
        self.transform = transform
        self.audio_files = [f for f in sorted(os.listdir(audio_dir)) if f.endswith('.wav')]
        self.labels = []
        for af in self.audio_files:
            parts = af.split('_')
            emotion = emotion_to_label(parts[2])
            intensity = intensity_to_label(parts[3].split('.')[0])
            self.labels.append((emotion, intensity))

    def __len__(self):
        """
        Returns the total number of audio files in the dataset.
        """
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Retrieves the audio file at the specified index.

        Args:
            idx (int): The index of the audio file to retrieve.

        Returns:
            The audio data as a tensor.
        """
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, self.labels[idx]

class CREMADDataset(Dataset):
    def __init__(self, video_dir, audio_dir, video_transform=None, audio_transform=None):
        """
        A dataset for loading both video and audio files from directories.

        Args:
            video_dir (str): Path to the directory containing video files.
            audio_dir (str): Path to the directory containing audio files.
            video_transform (callable, optional): Optional transform to be applied on video samples.
            audio_transform (callable, optional): Optional transform to be applied on audio samples.
        """
        self.video_dataset = CREMADVideo(video_dir, transform=video_transform)
        self.audio_dataset = CREMADAudio(audio_dir, transform=audio_transform)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return min(len(self.video_dataset), len(self.audio_dataset))

    def __getitem__(self, idx):
        """
        Retrieves the video and audio files at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            A tuple containing the video data, audio data, and their labels.
        """
        video_data, video_label = self.video_dataset[idx]
        audio_data, audio_label = self.audio_dataset[idx]
        if video_label != audio_label:
            print(f"Warning: Mismatched label at index {idx}: video label {video_label}, audio label {audio_label}")
            time.sleep(0.5)

        return (video_data, audio_data), video_label


if __name__ == '__main__':
    home = str(pathlib.Path.home())
    video_dir = home + "/data/crema-d-mirror/MP4Video"
    audio_dir = home + "/data/crema-d-mirror/AudioWAV"

    a_transform = transforms.Compose([
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, win_length=40, hop_length=160)
    ])

    crema_ds = CREMADDataset(video_dir, audio_dir, audio_transform=a_transform)
    print(f"Dataset size: {len(crema_ds)}")
    sample, label = crema_ds[0]
    print(f"Sample video shape: {sample[0].shape}, Sample audio shape: {sample[1].shape}, Label: {label}")
