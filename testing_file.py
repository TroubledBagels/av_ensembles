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
train_dir = home + "/data/av_shd_train"
test_dir = home + "/data/av_shd_test"

# Retrieve cached dataset saved at data_dir
av_ds_tr = SavedAVDataset(train_dir)
av_ds_te = SavedAVDataset(test_dir)

tr_dl =  torch.utils.data.DataLoader(av_ds_tr, batch_size=1, shuffle=True)
te_dl = torch.utils.data.DataLoader(av_ds_te, batch_size=1, shuffle=True)

model = AudioClassifier(1, 2)
print(model)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_fn = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm.tqdm(tr_dl)
    for batch_idx, (data, target) in enumerate(pbar):
        data = [d.float().to(device) for d in data]
        target = target.long().to(device)

        target_binary = torch.zeros(len(target), 2, device=device)
        mask_c1 = (target == model.c_1)
        mask_c2 = (target == model.c_2)
        target_binary[mask_c1, 0] = 1.0
        target_binary[mask_c2, 1] = 1.0

        optimizer.zero_grad()
        output = model(data[0])
        output = nn.LogSoftmax(dim=1)(output)
        loss = loss_fn(output, target_binary)
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
            if target != model.c_1 and target != model.c_2:
                continue
            data = [d.float().to(device) for d in data]
            target = target.long().to(device)
            output = model(data[0])
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            qbar.set_postfix(accuracy=correct / total)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")