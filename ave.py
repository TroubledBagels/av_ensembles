import torch
import torch.nn as nn
from datasets.AVE_DS import AVEAudio, make_mono, DSType, AVEVideo
from models.AVESquare import AudioClassifier, VideoClassifier
import pathlib
import os
import tqdm
from torchvision import transforms
import torchaudio

home = str(pathlib.Path.home())
data_dir = os.path.join(home, 'data', 'AVE_Dataset')

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

# av_ds = AVEAudio(ds_dir=data_dir, transform=a_transform, dstype=DSType.TRAIN_VAL)  # type=2 for train/val set
# av_ds_te = AVEAudio(ds_dir=data_dir, transform=a_transform, dstype=DSType.TEST)  # type=4 for test set
av_ds = AVEVideo(ds_dir=data_dir, dstype=DSType.TRAIN_VAL, transform=v_transform)  # type=2 for train/val set
av_ds_te = AVEVideo(ds_dir=data_dir, dstype=DSType.TEST, transform=v_transform)  # type=4 for test set
print(f"Dataset size: {len(av_ds)}")

classes = [0, 1]

av_ds.trim_classes(classes)
av_ds_te.trim_classes(classes)
av_dl = torch.utils.data.DataLoader(av_ds, batch_size=1, shuffle=True)
av_dl_te = torch.utils.data.DataLoader(av_ds_te, batch_size=1, shuffle=False)

model = VideoClassifier(classes[0], classes[1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
model.train()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10
for epoch in range(num_epochs):
    pbar = tqdm.tqdm(av_dl)
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.float().to(device)
        target = target.long().to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (batch_idx + 1), accuracy=100.0 * correct / total)
    avg_loss = total_loss / len(av_dl)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        qbar = tqdm.tqdm(av_dl_te)
        for data, target in qbar:
            data = data.float().to(device)
            target = target.long().to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            qbar.set_postfix(accuracy=100.0 * correct / total)
    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%")