from datasets.list_ds import AVDataset, SavedAVDataset
import tonic
import pathlib
from models.AVBSquare import AudioClassifier, VideoClassifier, AVBSquare, OutLayer
import torch
import torch.nn as nn
import snntorch as snn
from random import shuffle
import tqdm
import matplotlib.pyplot as plt

home = str(pathlib.Path.home())
train_dir = home + "/data/av_shd_train"
test_dir = home + "/data/av_shd_test"

# Retrieve cached dataset saved at data_dir
av_tr = SavedAVDataset(train_dir)
av_te = SavedAVDataset(test_dir)
print(av_tr[0][0][0].shape, av_tr[0][0][1].shape, av_tr[0][1])

model = AVBSquare()
print(model)

tr_ds_a = av_tr.get_a_ds()
tr_ds_v = av_tr.get_v_ds()
te_ds_a = av_te.get_a_ds()
te_ds_v = av_te.get_v_ds()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

model.load_state_dict(torch.load("./model_saves/av_bsquare.pth", map_location=device))

output_layer = OutLayer(200, 10).to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False
    param.requires_grad_(False)
optimizer = torch.optim.Adam(output_layer.parameters(), lr=1e-3)
te_dl = torch.utils.data.DataLoader(av_te, batch_size=1, shuffle=False)
tr_dl = torch.utils.data.DataLoader(av_tr, batch_size=1, shuffle=True)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    output_layer.train()
    pbar = tqdm.tqdm(tr_dl)
    total_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data = [d.float().to(device) for d in data]
        target = target.long().to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            _, out_mat = model(data[0], data[1])
            votes = out_mat.view(-1, 200).detach()
            # votes = nn.Softmax(dim=1)(votes)

        if hasattr(output_layer, "reset_mem"):
            output_layer.reset_mem()

        output, _ = output_layer(votes)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (batch_idx + 1))

    avg_loss = total_loss / len(tr_dl)
    print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")

    output_layer.eval()
    correct = 0
    total = 0
    pbar = tqdm.tqdm(te_dl)
    with torch.no_grad():
        for data, target in pbar:
            data = [d.float().to(device) for d in data]
            target = target.long().to(device)

            _, out_mat = model(data[0], data[1])
            out_mat = out_mat / out_mat.sum(dim=1, keepdim=True)
            votes = out_mat.view(-1, 10 * 10)
            output = output_layer(votes)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            pbar.set_postfix(accuracy=correct / total)
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/10, Test Accuracy: {accuracy:.2f}%")

# model.run_test_ds(av_te, device=device)

# acc_mat = model.retest_all_classifiers(te_ds_a, te_ds_v)
# Generate a heatmap from out_mat
# _, out_mat = model(torch.Tensor(te_ds_a[0][0]).unsqueeze(0).to(device), torch.Tensor(te_ds_v[0][0]).unsqueeze(0).to(device))
# acc_mat = out_mat.detach().cpu().squeeze().numpy()
# plt.imshow(acc_mat, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title("Classifier Accuracy Heatmap")
# # Add a note that says the top right side is audio only and bottom left is video only
# plt.text(0.5, -0.1, "Top Right: Audio Only | Bottom Left: Video Only", ha='center', va='center', transform=plt.gca().transAxes)
# plt.savefig("./model_saves/av_bsquare_test_heatmap.png")
# plt.show()
# model.train_classifiers(tr_ds_a, tr_ds_v, te_ds_a, te_ds_v, epochs=5, lr=1e-3, device=device)

# torch.save(model.state_dict(), f"./model_saves/av_bsquare.pth")
