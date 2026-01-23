from datasets.list_ds import AVDataset, SavedAVDataset
import tonic
import pathlib
from models.AVBSquare import AudioClassifier, VideoClassifier
import torch
import torch.nn as nn
import snntorch as snn
from random import shuffle
import tqdm

home = str(pathlib.Path.home())
data_dir = home + "/data/av_shd"

# Retrieve cached dataset saved at data_dir
av_ds = SavedAVDataset(data_dir)
print(av_ds[0][0][0].shape, av_ds[0][0][1].shape, av_ds[0][1])

model = VideoClassifier()
print(model)

av_ds.shuffle_data()
tr_ub = int(0.8 * len(av_ds))
tr_ds, te_ds = torch.utils.data.random_split(av_ds, [tr_ub, len(av_ds) - tr_ub])

tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=1, shuffle=True)
te_dl = torch.utils.data.DataLoader(te_ds, batch_size=1, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm.tqdm(tr_dl)
    for batch_idx, (data, target) in enumerate(pbar):
        data = [d.float().to(device) for d in data]
        target = target.long().to(device)

        optimizer.zero_grad()
        output = model(data[1])
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (batch_idx + 1))
    avg_loss = total_loss / len(tr_dl)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    qbar = tqdm.tqdm(te_dl)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in qbar:
            data = [d.float().to(device) for d in data]
            target = target.long().to(device)
            output = model(data[1])
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            qbar.set_postfix(accuracy=correct / total)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")