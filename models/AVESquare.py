import torch.nn as nn
import torch
import tqdm

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
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        # Input is (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)  # Merge batch and time
        x = self.conv_block(x)
        x = x.reshape(B, T, -1)  # Reshape for LSTM
        h0 = torch.zeros(2, B, 128).to(x.device)
        c0 = torch.zeros(2, B, 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class AVESquare(nn.Module):
    def __init__(self):
        super(AVESquare, self).__init__()
        self.num_classes = 10
        self.classifiers = nn.ModuleList()
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                self.classifiers.append(AudioClassifier(c_1=i, c_2=j))
                self.classifiers.append(VideoClassifier(c_1=i, c_2=j))

    def forward(self, x_audio, x_video):
        B_a, T_a, C_a, L_a = x_audio.size()
        B_v, T_v, C_v, H_v, W_v = x_video.size()
        assert B_a == B_v, "Batch sizes of audio and video must match"
        out_mat = torch.zeros(B_a, self.num_classes * 2, self.num_classes, device=x_audio.device)
        votes = torch.zeros(B_a, self.num_classes, device=x_audio.device)
        if x_audio.device == "cuda":
            torch.cuda.empty_cache()
            streams = []
            for _ in self.classifiers:
                streams.append(torch.cuda.Stream())

            for idx, classifier in enumerate(self.classifiers):
                with torch.cuda.stream(streams[idx]):
                    if isinstance(classifier, AudioClassifier):
                        out = classifier(x_audio)
                        # store in out_mat the output spike values for class c_1 at (c_1, c_2) and for c_2 at (c_1 + 1, c_2)
                        out_mat[:, 2 * classifier.c_1, classifier.c_2] = out[:, 0]
                        out_mat[:, 2 * classifier.c_1 + 1, classifier.c_2] = out[:, 1]
                    else:
                        out = classifier(x_video)
                        out_mat[:, 2 * classifier.c_2, classifier.c_1] = out[:, 1]
                        out_mat[:, 2 * classifier.c_2 + 1, classifier.c_1] = out[:, 0]
                    preds = out.argmax(dim=1)
                    c_1 = classifier.c_1
                    c_2 = classifier.c_2
                    for b in range(B_a):
                        if preds[b] == 0:
                            votes[b, c_1] += 1
                        else:
                            votes[b, c_2] += 1

            for s in streams:
                s.synchronize()

        else:
            for classifier in self.classifiers:
                if isinstance(classifier, AudioClassifier):
                    out = classifier(x_audio)
                    # store in out_mat the output spike values for class c_1 at (c_1, c_2) and for c_2 at (c_1 + 1, c_2)
                    out_mat[:, 2 * classifier.c_1, classifier.c_2] = out[:, 0]
                    out_mat[:, 2 * classifier.c_1 + 1, classifier.c_2] = out[:, 1]
                else:
                    out = classifier(x_video)
                    out_mat[:, 2 * classifier.c_2, classifier.c_1] = out[:, 1]
                    out_mat[:, 2 * classifier.c_2 + 1, classifier.c_1] = out[:, 0]
                preds = out.argmax(dim=1)
                c_1 = classifier.c_1
                c_2 = classifier.c_2
                for b in range(B_a):
                    if preds[b] == 0:
                        votes[b, c_1] += 1
                    else:
                        votes[b, c_2] += 1

        return votes, out_mat

    def train_classifiers(self, tr_ds_a, tr_ds_v, te_ds_a, te_ds_v, epochs=10, lr=1e-3, device='cpu'):
        optimisers = []
        loss_fn = nn.KLDivLoss(reduction='batchmean')
        for c_idx, classifier in enumerate(self.classifiers):
            optimisers.append(torch.optim.Adam(classifier.parameters(), lr=lr))
            if type(classifier) == AudioClassifier:
                print(f"Training Audio Classifier for classes {classifier.c_1} and {classifier.c_2}")
            else:
                print(f"Training Video Classifier for classes {classifier.c_1} and {classifier.c_2}")

            for epoch in range(epochs):
                classifier.train()
                total_loss = 0
                correct = 0
                total = 0
                if type(classifier) == AudioClassifier:
                    tr_dl = torch.utils.data.DataLoader(tr_ds_a, batch_size=1, shuffle=True)
                    te_dl = torch.utils.data.DataLoader(te_ds_a, batch_size=1, shuffle=True)
                else:
                    tr_dl = torch.utils.data.DataLoader(tr_ds_v, batch_size=4, shuffle=True)
                    te_dl = torch.utils.data.DataLoader(te_ds_v, batch_size=1, shuffle=True)

                pbar = tqdm.tqdm(tr_dl)
                for batch_idx, (data, target) in enumerate(pbar):
                    data = data.float().to(device)
                    target = target.long().to(device)

                    target_binary = torch.zeros(len(target), 2).to(device)
                    mask_c1 = (target == classifier.c_1)
                    mask_c2 = (target == classifier.c_2)
                    target_binary[mask_c1, 0] = 1.0
                    target_binary[mask_c2, 1] = 1.0

                    optimisers[c_idx].zero_grad()
                    output = classifier(data)
                    output = nn.LogSoftmax(dim=1)(output)
                    loss = loss_fn(output, target_binary)
                    loss.backward()
                    optimisers[c_idx].step()

                    total_loss += loss.item()
                    pbar.set_postfix(loss=total_loss / (batch_idx + 1))
                avg_loss = total_loss / len(tr_dl)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

                classifier.eval()
                correct = 0
                total = 0
                qbar = tqdm.tqdm(te_dl)
                with (torch.no_grad()):
                    for data, target in qbar:
                        if target != classifier.c_1 and target != classifier.c_2:
                            continue
                        elif target == classifier.c_1:
                            target = torch.zeros_like(target)
                        else:
                            target = torch.ones_like(target)
                        data = data.float().to(device)
                        target = target.long().to(device)
                        output = classifier(data)
                        preds = output.argmax(dim=1)
                        correct += (preds == target).sum().item()
                        total += target.size(0)
                        qbar.set_postfix(accuracy=correct / total)
                accuracy = 100 * correct / total
                print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%")

    def retest_all_classifiers(self, te_ds_a, te_ds_v, device='cpu'):
        max_class = -1
        for classifier in self.classifiers:
            if classifier.c_1 > max_class:
                max_class = classifier.c_1
            if classifier.c_2 > max_class:
                max_class = classifier.c_2
        num_classes = max_class + 1

        acc_mat = torch.zeros(num_classes, num_classes)
        for classifier in self.classifiers:
            correct = 0
            total = 0
            if type(classifier) == AudioClassifier:
                te_dl = torch.utils.data.DataLoader(te_ds_a, batch_size=1, shuffle=True)
            else:
                te_dl = torch.utils.data.DataLoader(te_ds_v, batch_size=1, shuffle=True)
            with torch.no_grad():
                for data, target in te_dl:
                    if target != classifier.c_1 and target != classifier.c_2:
                        continue
                    elif target == classifier.c_1:
                        target = torch.zeros_like(target)
                    else:
                        target = torch.ones_like(target)
                    data = data.float().to(device)
                    target = target.long().to(device)
                    output = classifier(data)
                    preds = output.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)
            accuracy = 100 * correct / total
            acc_mat[classifier.c_1, classifier.c_2] = accuracy
            acc_mat[classifier.c_2, classifier.c_1] = accuracy
            print(f"Classifier for classes {classifier.c_1} and {classifier.c_2} Accuracy: {accuracy:.2f}%")
        return acc_mat

    def run_test_ds_vote(self, te_ds, device='cpu'):
        te_dl = torch.utils.data.DataLoader(te_ds, batch_size=1, shuffle=True)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in te_dl:
                data = [d.float().to(device) for d in data]
                target = target.long().to(device)
                output, _ = self(data[0], data[1])
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)
        accuracy = 100 * correct / total
        print(f"Test Dataset Voting Accuracy: {accuracy:.2f}%")
        return accuracy

if __name__ == '__main__':
    model = VideoClassifier(0, 1)
    print(model)
    sample_input = torch.randn(4, 64, 3, 224, 224)
    output = model(sample_input)
    print(output.shape)  # Expected output shape: (4, 2)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")