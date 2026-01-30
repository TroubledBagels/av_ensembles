import torch.nn as nn
import torch


class AudioClassifier(nn.Module):
    def __init__(self, c_1, c_2):
        # Interpreting a MelSpectrogram of size (1, 128, T)
        super(AudioClassifier, self).__init__()
        self.c_1 = c_1
        self.c_2 = c_2

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.lstm = nn.LSTM(input_size=48, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        # mel-spectrogram input shape: (B, C, 128, T)
        # vectorise to (B, T, C*F)
        x = self.conv_block(x)
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # (B, T, C*F)

        h0 = torch.zeros(2, x.size(0), 128).to(x.device)
        c0 = torch.zeros(2, x.size(0), 128).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)

        return out

class VideoClassifier(nn.Module):
    def __init__(self, c_1, c_2):
        # Placeholder for VideoClassifier
        super(VideoClassifier, self).__init__()
        self.c_1 = c_1
        self.c_2 = c_2
        # Define layers here as needed
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        # Input is (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)  # Merge batch and time
        x = self.conv_block(x)
        print(x.shape)
        print(B, T, C, H, W)
        x = x.reshape(B, T, -1)  # Reshape for LSTM
        h0 = torch.zeros(2, B, 128).to(x.device)
        c0 = torch.zeros(2, B, 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

if __name__ == '__main__':
    model = VideoClassifier(0, 1)
    print(model)
    sample_input = torch.randn(4, 64, 3, 224, 224)
    output = model(sample_input)
    print(output.shape)  # Expected output shape: (4, 2)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")