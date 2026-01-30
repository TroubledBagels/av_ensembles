import torch
import torch.nn as nn
import pathlib
import tqdm

class AudioClassifier(nn.Module):
    def __init__(self, c_1, c_2, input_size=96, hidden_size=64, num_layers=2):
        super(AudioClassifier, self).__init__()
        self.c_1 = c_1
        self.c_2 = c_2
        conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, stride=2, padding=1)
        pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.conv_block = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool,
            conv2,
            nn.ReLU(),
            pool
        )
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # mel-spectrogram input shape: (B, C, 80, T)
        print(x.shape)
        x = self.conv_block(x)
        print(x.shape)
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.contiguous().view(x.size(0), x.size(1), -1) # (B, T, C*F)
        print(x.shape)
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class VideoClassifier(nn.Module):
    def __init__(self, c_1, c_2, input_size=128, hidden_size=64, num_layers=2):
        super(VideoClassifier, self).__init__()
        self.c_1 = c_1
        self.c_2 = c_2
        conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=3, padding=1)
        pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1)
        conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_block = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool,
            conv2,
            nn.ReLU(),
            pool,
            conv3,
            nn.ReLU(),
            # pool,
            # conv4,
            # nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        B, T, C, H, W = x.size() # C is 3 for RGB
        x = x.reshape(B*T, C, H, W)  # (B*T, C, H, W)
        f = self.conv_block(x).flatten(1) # (B*T, 32)
        x = f.view(B, T, -1)  # (B, T, 32)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


if __name__ == '__main__':
    audio_model = AudioClassifier(c_1=0, c_2=1)
    print(audio_model)

    video_model = VideoClassifier(c_1=0, c_2=1)
    print(video_model)

    sample_audio = torch.randn(1, 1, 80, 3641) # Batch size 4, 1 channel, 128 length
    audio_output = audio_model(sample_audio)
    print("Audio output shape:", audio_output.shape)
    print(f"Parameters in Audio Model: {sum(p.numel() for p in audio_model.parameters() if p.requires_grad)}")
    sample_video = torch.randn(1, 67, 3, 120, 160) # Batch size 4, 60 frames, 360x480 RGB
    video_output = video_model(sample_video)
    print("Video output shape:", video_output.shape)
    print(f"Parameters in Video Model: {sum(p.numel() for p in video_model.parameters() if p.requires_grad)}")

