import torch
from torch.utils.data import Dataset
from random import shuffle
import torchvision.transforms as transforms
import torchaudio.transforms as T
import os
import pathlib
import torchvision
import torchaudio

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
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

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

        if self.transform:
            video_data = self.transform(video_data)

        return video_data, self.video_files[idx]

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
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

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

        return waveform, self.audio_files[idx]

class ListDataset(Dataset):
    def __init__(self, data_list):
        """
        A simple dataset that wraps a list of data points and labels, i.e. (data, label).

        Args:
            data_list (list): A list containing the data points.
        """
        self.data_list = data_list

    def __len__(self):
        """
        Returns the total number of data points in the dataset.
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Retrieves the data point at the specified index.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            The data point at the specified index.
        """
        return self.data_list[idx]

    # Define a method to sort the dataset based on labels
    def sort(self, key_func):
        """
        Sorts the dataset based on a key function applied to each data point.

        Args:
            key_func (function): A function that takes a data point and returns a value to sort by.
        """
        self.data_list = sorted(self.data_list, key=key_func)


class AVDataset(Dataset):
    def __init__(self, a_ds, v_ds, v=0):
        """
        Constructs a dataset that pairs audio and video points from two separate datasets with matching labels.
        :param a_ds:
        :param v_ds:
        :param max_pairs:
        """
        self.verbose = v
        self.data = []
        self.a_ds = a_ds
        self.v_ds = v_ds
        self._create_pairs()

    def _create_pairs(self):
        # Match up audio and video samples based on labels and store them in data
        v_label_indices = {}
        for idx in range(len(self.v_ds)):
            _, label = self.v_ds[idx]
            if label not in v_label_indices:
                v_label_indices[label] = []
            v_label_indices[label].append(idx)
            if self.verbose > 0 and idx % 1000 == 0:
                print(f"\rProcessed {idx/len(self.v_ds)} video samples", end="")
        if self.verbose > 0:
            print()
        for a_idx in range(len(self.a_ds)):
            a_data, a_label = self.a_ds[a_idx]
            if a_label in v_label_indices and len(v_label_indices[a_label]) > 0:
                v_idx = v_label_indices[a_label].pop(0)
                v_data, v_label = self.v_ds[v_idx]
                self.data.append(((a_data, v_data), a_label))
            if self.verbose > 0 and a_idx % 1000 == 0:
                print(f"\rPaired {a_idx/len(self.a_ds)} audio samples", end="")
        if self.verbose > 0:
            print()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

class SavedAVDataset(Dataset):
    def __init__(self, save_loc):
        self.save_loc = save_loc
        self.files = sorted(
            f for f in os.listdir(save_loc) if f.endswith('.pt')
        )
        # Load into memory
        self.data = []
        for f in self.files:
            file_path = os.path.join(self.save_loc, f)
            self.data.append(torch.load(file_path, weights_only=False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def shuffle_data(self):
        shuffle(self.data)

    def get_a_ds(self):
        a_data = []
        for (a, v), label in self.data:
            a_data.append((a, label))
        return ListDataset(a_data)

    def get_v_ds(self):
        v_data = []
        for (a, v), label in self.data:
            v_data.append((v, label))
        return ListDataset(v_data)

def create_av_shd_dataset():
    import tonic
    from tonic.datasets import hsd, nmnist

    home = str(pathlib.Path.home())
    ds_path_hsd = home + "/data/shd"
    ds_path_nmnist = home + "/data/nmnist"

    v_transform = tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=10000),
    ])

    a_transform = tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=tonic.datasets.SHD.sensor_size, time_window=10000),
    ])

    a_ds = hsd.SHD(save_to=ds_path_hsd, train=True, transform=a_transform)  # In form (data, label), with data being (time, neuron, p)
    label_set = set()
    label_counts = {}
    german_a_ds = []
    for data, label in a_ds:
        if label >= 10:
            german_a_ds.append((data, label - 10))
            if label-10 not in label_counts:
                label_counts[label-10] = 0
            label_counts[label-10] += 1
            label_set.add(label - 10)
    a_ds = ListDataset(german_a_ds)
    print(f"Audio dataset labels: {label_set}")
    print(f"Audio dataset label counts: {label_counts}")
    print(f"Audio dataset length: {len(a_ds)}")

    v_ds = nmnist.NMNIST(save_to=ds_path_nmnist, train=False, transform=v_transform)
    v_ds = ListDataset(v_ds)
    v_ds.sort(key_func=lambda x: x[1])  # Sort by label
    print(f"Video dataset length: {len(v_ds)}")

    label_indices = {}
    for idx in range(len(v_ds)):
        _, label = v_ds[idx]
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)
        if idx % 1000 == 0:
            print(f"\rProcessed {idx/len(v_ds)} video samples", end="")
    print()
    print(f"Video dataset label indices: {{label: number of samples}}")

    # Randomly sample without replacement to match audio dataset counts
    balanced_v_indices = []
    for label in label_counts:
        indices = label_indices[label]
        shuffle(indices)
        balanced_v_indices += indices[:label_counts[label]]
        print(f"Label {label}: selected {len(indices[:label_counts[label]])} samples")
    balanced_v_ds = ListDataset([v_ds[idx] for idx in balanced_v_indices])
    v_ds = balanced_v_ds

    av_ds = AVDataset(a_ds, v_ds)
    print(f"AV dataset length: {len(av_ds)}")

    save_loc = home + "/data/av_shd_train"
    # Save each piece of data separately for easy loading later
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    for idx in range(len(av_ds)):
        data_point = av_ds[idx]
        torch.save(data_point, os.path.join(save_loc, f"av_data_{idx}.pt"))
        if idx % 1000 == 0:
            print(f"\rSaved {idx/len(av_ds)} AV data samples", end="")
    print()
    print(f"Saved AV dataset to {save_loc}")

def save_crema_d_video_tensors():
    home = str(pathlib.Path.home())
    video_loc = pathlib.Path.home() / "data" / "crema-d-mirror" / "MP4Video"
    crema_ds = CREMADVideo(str(video_loc))
    print(f"CREMA-D Video Dataset size: {len(crema_ds)}")

    save_loc = home + "/data/CREMA_D/video_tensors"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    for idx in range(len(crema_ds)):
        video_data, video_name = crema_ds[idx]
        torch.save(video_data, os.path.join(save_loc, f"{video_name.split('.')[0]}.pt"))
        if (idx + 1) % 100 == 0 or idx == len(crema_ds)-1:
            print(f"\rSaved {idx+1}/{len(crema_ds)} video tensors", end="")
    print()
    print(f"Saved CREMA-D video tensors to {save_loc}")

def save_crema_d_audio_tensors():
    a_transform = transforms.Compose([
        T.MelSpectrogram(sample_rate=16000, n_mels=64)
    ])

    home = str(pathlib.Path.home())
    audio_loc = pathlib.Path.home() / "data" / "crema-d-mirror" / "AudioWAV"
    crema_ds = CREMADAudio(str(audio_loc), transform=a_transform)
    print(f"CREMA-D Audio Dataset size: {len(crema_ds)}")
    save_loc = home + "/data/CREMA_D/audio_tensors"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    for idx in range(len(crema_ds)):
        audio_data, audio_name = crema_ds[idx]
        torch.save(audio_data, os.path.join(save_loc, f"{audio_name.split('.')[0]}.pt"))
        if (idx + 1) % 100 == 0 or idx == len(crema_ds)-1:
            print(f"\rSaved {idx+1}/{len(crema_ds)} audio tensors", end="")
    print()
    print(f"Saved CREMA-D audio tensors to {save_loc}")

if __name__ == "__main__":
    # Uncomment the following lines to create datasets
    # create_av_shd_dataset()
    # save_crema_d_video_tensors()
    # save_crema_d_audio_tensors()
    pass

