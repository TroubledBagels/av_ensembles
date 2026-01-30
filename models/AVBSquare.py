import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate as sur
import tqdm
import numpy as np

class AudioClassifier(nn.Module):
    def __init__(self, c_1, c_2):
        super(AudioClassifier, self).__init__()
        self.c_1 = c_1
        self.c_2 = c_2

        # Audio starts as 1D signal of (t, 1, 700)
        self.sur_grad = sur.atan()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=13, padding=1, stride=5) # Output: (16, 139)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.sur_grad)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=1, stride=3) # (32, 22)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=self.sur_grad)
        self.fc1 = nn.Linear(32 * 22, 32)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=self.sur_grad)
        self.fc2 = nn.Linear(32, 2)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=self.sur_grad)

    def forward(self, x, train=False):
        B, T, C, L = x.size()  # B: batch size, T: time steps, C: channels, L: length
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        outs = []
        for t in range(T):
            x_t = x[:, t, :, :].squeeze(2)  # Shape: (B, L)
            x_t = self.conv1(x_t)
            spk1, mem1 = self.lif1(x_t, mem1)
            x_t = self.pool(spk1)
            x_t = self.conv2(x_t)
            spk2, mem2 = self.lif2(x_t, mem2)
            x_t = spk2.view(B, -1)
            x_t = self.fc1(x_t)
            spk3, mem3 = self.lif3(x_t, mem3)
            x_t = self.fc2(spk3)
            spk4, mem4 = self.lif4(x_t, mem4)
            if train:
                outs.append(mem4)
            else:
                outs.append(spk4)
        if train:
            x = torch.stack(outs, dim=0).mean(dim=0)  # Average over time steps
        else:
            x = torch.stack(outs, dim=0).sum(dim=0)  # Sum over time steps
        return x

class VideoClassifier(nn.Module):
    def __init__(self, c_1, c_2):
        super(VideoClassifier, self).__init__()
        self.c_1 = c_1
        self.c_2 = c_2

        # Video starts as 2D signal of (t, 2, 34, 34)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=7, padding=2, stride=3) # Output: (16, 11, 11)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=sur.atan())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=2, stride=3) # Output: (32, 1, 1)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=sur.atan())
        self.fc1 = nn.Linear(32 * 1 * 1, 32)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=sur.atan())
        self.fc2 = nn.Linear(32, 2)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=sur.atan())

    def forward(self, x, train=False):
        B, T, C, H, W = x.size()  # B: batch size, T: time steps, C: channels, H: height, W: width
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        outs = []
        for t in range(T):
            x_t = x[:, t, :, :, :]  # Shape: (B, C, H, W)
            x_t = self.conv1(x_t)
            spk1, mem1 = self.lif1(x_t, mem1)
            x_t = self.pool(spk1)
            x_t = self.conv2(x_t)
            spk2, mem2 = self.lif2(x_t, mem2)
            x_t = spk2.view(B, -1)
            x_t = self.fc1(x_t)
            spk3, mem3 = self.lif3(x_t, mem3)
            x_t = self.fc2(spk3)
            spk4, mem4 = self.lif4(x_t, mem4)
            if train:
                outs.append(mem4)
            else:
                outs.append(spk4)
        if train:
            x = torch.stack(outs, dim=0).mean(dim=0)  # Average over time steps
        else:
            x = torch.stack(outs, dim=0).sum(dim=0)  # Sum over time steps
        return x


class AVBSquare(nn.Module):
    def __init__(self):
        super(AVBSquare, self).__init__()
        self.num_classes = 10
        self.classifiers = nn.ModuleList()
        for i in range(self.num_classes):
            for j in range(i+1, self.num_classes):
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
                        out_mat[:, 2*classifier.c_1, classifier.c_2] = out[:, 0]
                        out_mat[:, 2*classifier.c_1 + 1, classifier.c_2] = out[:, 1]
                    else:
                        out = classifier(x_video)
                        out_mat[:, 2*classifier.c_2, classifier.c_1] = out[:, 1]
                        out_mat[:, 2*classifier.c_2 + 1, classifier.c_1] = out[:, 0]
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

    def train_classifiers(self, tr_ds_a, tr_ds_v, te_ds_a, te_ds_v, epochs=10, lr=1e-3, device=torch.device("cpu")):
        optimisers = []
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        for c_idx, classifier in enumerate(self.classifiers):
            optimisers.append(torch.optim.Adam(classifier.parameters(), lr=lr))
            if type(classifier) == AudioClassifier:
                print(f"Training Audio Classifier {c_idx+1}/{len(self.classifiers)} for classes {classifier.c_1} vs {classifier.c_2}")
            else:
                print(f"Training Video Classifier {c_idx+1}/{len(self.classifiers)} for classes {classifier.c_1} vs {classifier.c_2}")

            for epoch in range(epochs):
                classifier.train()
                total_loss = 0
                if type(classifier) == AudioClassifier:
                    tr_dl = torch.utils.data.DataLoader(tr_ds_a, batch_size=1, shuffle=True)
                    te_dl = torch.utils.data.DataLoader(te_ds_a, batch_size=1, shuffle=False)
                else:
                    tr_dl = torch.utils.data.DataLoader(tr_ds_v, batch_size=1, shuffle=True)
                    te_dl = torch.utils.data.DataLoader(te_ds_v, batch_size=1, shuffle=False)

                pbar = tqdm.tqdm(tr_dl)
                for batch_idx, (data, target) in enumerate(pbar):
                    data = data.float().to(device)
                    target = target.long().to(device)

                    target_binary = torch.zeros(len(target), 2, device=device)
                    mask_c1 = (target == classifier.c_1)
                    mask_c2 = (target == classifier.c_2)
                    target_binary[mask_c1, 0] = 1.0
                    target_binary[mask_c2, 1] = 1.0

                    optimisers[c_idx].zero_grad()
                    output = classifier(data, True)
                    output = nn.LogSoftmax(dim=1)(output)
                    loss = loss_fn(output, target_binary)
                    loss.backward()
                    optimisers[c_idx].step()

                    total_loss += loss.item()
                    pbar.set_postfix(loss=total_loss / (batch_idx + 1))
                avg_loss = total_loss / len(tr_dl)
                print(f"Classifier {c_idx+1}/{len(self.classifiers)} Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

                qbar = tqdm.tqdm(te_dl)
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in qbar:
                        if target != classifier.c_1 and target != classifier.c_2:
                            continue
                        data = data.float().to(device)
                        target = target.long().to(device)
                        output = classifier(data)
                        preds = output.argmax(dim=1)
                        correct += (preds == target).sum().item()
                        total += target.size(0)
                        qbar.set_postfix(acc=correct / total)
                accuracy = 100 * correct / total
                print(f"Classifier {c_idx+1}/{len(self.classifiers)} Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%")

    def retest_all_classifiers(self, te_ds_a, te_ds_v, device=torch.device("cpu")):
        max_class = -1
        for classifier in self.classifiers:
            if classifier.c_1 > max_class:
                max_class = classifier.c_1
            if classifier.c_2 > max_class:
                max_class = classifier.c_2
        max_class += 1

        acc_mat = np.zeros((max_class, max_class))

        for c_idx, classifier in enumerate(self.classifiers):
            if type(classifier) == AudioClassifier:
                print(f"Retesting Audio Classifier {c_idx+1}/{len(self.classifiers)} for classes {classifier.c_1} vs {classifier.c_2}")
                te_dl = torch.utils.data.DataLoader(te_ds_a, batch_size=1, shuffle=False)
            else:
                print(f"Retesting Video Classifier {c_idx+1}/{len(self.classifiers)} for classes {classifier.c_1} vs {classifier.c_2}")
                te_dl = torch.utils.data.DataLoader(te_ds_v, batch_size=1, shuffle=False)

            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in te_dl:
                    if target != classifier.c_1 and target != classifier.c_2:
                        continue
                    data = data.float().to(device)
                    target = target.long().to(device)
                    output = classifier(data)
                    preds = output.argmax(dim=1)
                    if preds == 0:
                        preds = classifier.c_1
                    else:
                        preds = classifier.c_2
                    correct += (preds == target).sum().item()
                    total += target.size(0)
            accuracy = 100 * correct / total
            print(f"Classifier {c_idx+1}/{len(self.classifiers)} Test Accuracy: {accuracy:.2f}%")
            if type(classifier) == AudioClassifier:
                acc_mat[classifier.c_1, classifier.c_2] = accuracy
            else:
                acc_mat[classifier.c_2, classifier.c_1] = accuracy
        return acc_mat

    def run_test_ds_vote(self, te_ds, device=torch.device("cpu")):
        te_dl = torch.utils.data.DataLoader(te_ds, batch_size=1, shuffle=False)
        correct = 0
        total = 0
        self.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(te_dl)
            for data, target in pbar:
                data_a = data[0].float().to(device)
                data_v = data[1].float().to(device)
                target = target.long().to(device)
                votes, _ = self(data_a, data_v)
                preds = votes.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)
                pbar.set_postfix(accuracy=correct / total)
        accuracy = 100 * correct / total
        print(f"Overall Test Accuracy: {accuracy:.2f}%")

class OutLayer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(OutLayer, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        # self.lif = snn.Leaky(beta=0.9, spike_grad=sur.atan())

    def forward(self, x):
        x = self.fc(x)
        return x, None
        # x, mem = self.lif(x)
        # return x, mem

    # def reset_mem(self):
    #     self.lif.reset_mem()

if __name__ == '__main__':
    model = AudioClassifier(0, 1)
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    dummy_input = torch.randn(4, 100, 1, 700)  # Batch size of 4, 100 time steps, 1 channel, length 700
    output = model(dummy_input)
    print(output.shape)

    model_v = VideoClassifier(0, 1)
    print(model_v)
    print(f"Number of parameters: {sum(p.numel() for p in model_v.parameters() if p.requires_grad)}")
    dummy_input_v = torch.randn(4, 100, 2, 34, 34)  # Batch size of 4, 100 time steps, 2 channels, 34x34
    output_v = model_v(dummy_input_v)
    print(output_v.shape)

    model = AVBSquare()
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    dummy_input_a = torch.randn(4, 100, 1, 700)
    dummy_input_v = torch.randn(4, 100, 2, 34, 34)
    output = model(dummy_input_a, dummy_input_v)
    print(output[0].shape)  # Votes
    print(output[1].shape)  # Out mat
    print(output[1])
    # comment
