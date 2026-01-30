import os
import torch
from torch.utils.data import Dataset
import torchvision
import torchaudio
import pathlib
import time
from torchvision import transforms
from enum import Enum

class DSType(Enum):
    ALL = 0
    TRAIN = 1
    VAL = 2
    TRAIN_VAL = 3
    TEST = 4

def label_from_string(label_str):
    label_dict = {
        'Accordion': 0,
        'Acoustic guitar': 1,
        'Baby cry, infant cry': 2,
        'Banjo': 3,
        'Bark': 4,
        'Bus': 5,
        'Cat': 6,
        'Chainsaw': 7,
        'Church bell': 8,
        'Clock': 9,
        'Female speech, woman speaking': 10,
        'Fixed-wing aircraft, airplane': 11,
        'Flute': 12,
        'Frying (food)': 13,
        'Goat': 14,
        'Helicopter': 15,
        'Horse': 16,
        'Male speech, man speaking': 17,
        'Mandolin': 18,
        'Motorcycle': 19,
        'Race car, auto racing': 20,
        'Rodents, rats, mice': 21,
        'Shofar': 22,
        'Toilet flush': 23,
        'Train horn': 24,
        'Truck': 25,
        'Ukulele': 26,
        'Violin, fiddle': 27
        # Add more mappings as needed
    }
    return label_dict.get(label_str, -1)

class AVEVideo(Dataset):
    def __init__(self, ds_dir, transform=None, dstype: DSType=DSType.ALL):
        """
        A dataset for loading video files from a directory.

        Args:
            video_dir (str): Path to the directory containing video files.
            transform (callable, optional): Optional transform to be applied on a sample.
            type (int): type of datasets:
                0: Entire set
                1: Train set
                2: Val set
                3: Train + Val set
                4: Test set
        """
        self.ds_dir = ds_dir
        self.transform = transform
        self.video_dir = os.path.join(self.ds_dir, 'AVE')
        txt_file = ""
        if dstype == DSType.ALL:
            txt_file = os.path.join(self.ds_dir, 'Annotations.txt')
        elif dstype == DSType.TRAIN:
            txt_file = os.path.join(self.ds_dir, 'trainSet.txt')
        elif dstype == DSType.VAL:
            txt_file = os.path.join(self.ds_dir, 'valSet.txt')
        elif dstype == DSType.TRAIN_VAL:
            txt_file = os.path.join(self.ds_dir, 'trainValSet.txt')
        elif dstype == DSType.TEST:
            txt_file = os.path.join(self.ds_dir, 'testSet.txt')
        else:
            raise ValueError("Invalid dataset type specified.")
        with open(txt_file, 'r') as f:
            # Each line is class&filename&quality&start_time&end_time
            lines = f.readlines()
        valid_files = []
        self.labels = []
        lines = sorted(lines, key=lambda x: x.split('&')[1])
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            parts = line.strip().split('&')
            if len(parts) >= 2:
                filename = parts[1]
                # Check filename + ".mp4" exists in audio directory
                if os.path.exists(
                        os.path.join(self.video_dir, filename + ".mp4")) and filename + ".mp4" not in valid_files:
                    valid_files.append(filename + ".mp4")
                    self.labels.append(label_from_string(parts[0]))
                else:
                    print(f"Video file {filename} does not exist.")
                    print("MISSING:", repr(filename + ".mp4"))
                    print("PATH:", repr(os.path.join(self.video_dir, filename + ".mp4")))
        self.video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.mp4') and f in valid_files]
        print(len(self.video_files), "video files found.")
        print(len(valid_files), "valid files found.")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # print(f"Getting item {idx}")
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        video, _, info = torchvision.io.read_video(video_path, pts_unit='sec')
        # print(f"Loaded video shape: {video.shape}, info: {info}")
        label = self.labels[idx]

        T = video.shape[0]
        idx_t = torch.linspace(0, T - 1, steps=64).long()
        video = video[idx_t]

        video = video.permute(0, 3, 1, 2).float() / 255.0
        video = torch.nn.functional.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)

        if self.transform:
            video = self.transform(video)

        return video, label

    def trim_classes(self, class_list):
        """
        Trims the dataset to only include samples with the specified classes.

        Args:
            class_list (list): The list of classes to keep.
        """
        filtered_indices = [i for i, label in enumerate(self.labels) if label in class_list]
        self.video_files = [self.video_files[i] for i in filtered_indices]
        # Renormalize labels to start from 0
        self.labels = [self.labels[i] for i in filtered_indices]
        self.labels = [class_list.index(label) for label in self.labels]

class AVEAudio(Dataset):
    def __init__(self, ds_dir, transform=None, dstype: DSType=DSType.ALL):
        """
        A dataset for loading audio files from a directory.

        Args:
            audio_dir (str): Path to the directory containing audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
            type (int): type of datasets:
                0: Entire set
                1: Train set
                2: Val set
                3: Train + Val set
                4: Test set
        """
        self.ds_dir = ds_dir
        self.transform = transform
        self.audio_dir = os.path.join(self.ds_dir, 'AVEAudio')
        txt_file = ""
        if dstype == DSType.ALL:
            txt_file = os.path.join(self.ds_dir, 'Annotations.txt')
        elif dstype == DSType.TRAIN:
            txt_file = os.path.join(self.ds_dir, 'trainSet.txt')
        elif dstype == DSType.VAL:
            txt_file = os.path.join(self.ds_dir, 'valSet.txt')
        elif dstype == DSType.TRAIN_VAL:
            txt_file = os.path.join(self.ds_dir, 'trainValSet.txt')
        elif dstype == DSType.TEST:
            txt_file = os.path.join(self.ds_dir, 'testSet.txt')
        else:
            raise ValueError("Invalid dataset type specified.")
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        valid_files = []
        self.labels = []
        temp_set = set()
        lines = sorted(lines, key=lambda x: x.split('&')[1])
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            parts = line.strip().split('&')
            if len(parts) >= 2:
                filename = parts[1]
                # Check filename + ".wav" exists in audio directory
                if os.path.exists(os.path.join(self.audio_dir, filename + ".wav")) and filename + ".wav" not in valid_files:
                    valid_files.append(filename + ".wav")
                    self.labels.append(label_from_string(parts[0]))
                else:
                    print(f"Audio file {filename} does not exist.")
                    print("MISSING:", repr(filename+".wav"))
                    print("PATH:", repr(os.path.join(self.audio_dir, filename + ".wav")))
        self.audio_files = [f for f in sorted(os.listdir(self.audio_dir)) if f.endswith('.wav') and f in valid_files]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)
        label = self.labels[idx]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

    def trim_classes(self, class_list):
        """
        Trims the dataset to only include samples with the specified classes.

        Args:
            class_list (list): The list of classes to keep.
        """
        filtered_indices = [i for i, label in enumerate(self.labels) if label in class_list]
        self.audio_files = [self.audio_files[i] for i in filtered_indices]
        # Renormalize labels to start from 0
        self.labels = [self.labels[i] for i in filtered_indices]
        self.labels = [class_list.index(label) for label in self.labels]

class AVEMulti(Dataset):
    def __init__(self, ds_dir, video_transform=None, audio_transform=None, dstype: DSType=DSType.ALL):
        """
        A dataset for loading both video and audio files from directories.

        Args:
            video_dir (str): Path to the directory containing video files.
            audio_dir (str): Path to the directory containing audio files.
            video_transform (callable, optional): Optional transform to be applied on video samples.
            audio_transform (callable, optional): Optional transform to be applied on audio samples.
            type (int): type of datasets:
                0: Entire set
                1: Train set
                2: Val set
                3: Train + Val set
                4: Test set
        """
        self.video_dataset = AVEVideo(ds_dir, transform=video_transform, dstype=dstype)
        self.audio_dataset = AVEAudio(ds_dir, transform=audio_transform, dstype=dstype)
        assert len(self.video_dataset) == len(self.audio_dataset), "Video and Audio datasets must be of the same length."

    def __len__(self):
        return len(self.video_dataset)

    def __getitem__(self, idx):
        video, label_v = self.video_dataset[idx]
        audio, label_a = self.audio_dataset[idx]
        assert label_v == label_a, "Labels for video and audio do not match."
        return (video, audio), label_v

def make_mono(tensor):
    if tensor.shape[0] > 1:
        tensor = torch.mean(tensor, dim=0, keepdim=True)
    return tensor

if __name__ == '__main__':
    v_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
    ])
    a_transform = transforms.Compose([
        torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000),
        make_mono,
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512),
        torchaudio.transforms.AmplitudeToDB()
    ])
    home_dir = pathlib.Path.home()
    data_dir = home_dir / "data" / "AVE_Dataset"
    dataset = AVEVideo(ds_dir=data_dir, dstype=DSType.TRAIN, transform=v_transform)
    print(f"Dataset size: {len(dataset)}")
    sample_video, sample_label = dataset[0]
    print(f"Sample video shape: {sample_video.shape}, Sample label: {sample_label}")