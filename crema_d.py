import torch
import torch.nn as nn
import datasets.CREMA_D_DS as cds
import datasets.list_ds as lds
import models.CREMASquare as CS
import pathlib
from torch.utils.data import DataLoader
import tqdm
from torchvision import transforms

home = str(pathlib.Path.home())

v_ds = cds.CREMADVideo(home + "/data/crema-d-mirror/MP4Video")
# subset = torch.utils.data.Subset(v_ds, list(range(4)))
v_ds.trim_classes(0, 1)

v_dl = DataLoader(v_ds, batch_size=16, shuffle=True)
print(f"Dataset size: {len(v_ds)}")

model = CS.VideoClassifier(0, 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

model.train()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0
    pbar = tqdm.tqdm(v_dl)
    for batch_idx, (data, (target, _)) in enumerate(pbar):
        data = (data.float() / 255.0).to(device)
        target = target.long().to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (batch_idx + 1))
    avg_loss = total_loss / len(v_ds)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")