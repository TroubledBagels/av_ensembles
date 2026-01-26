from datasets.list_ds import AVDataset, SavedAVDataset
import tonic
import pathlib
from models.AVBSquare import AudioClassifier, VideoClassifier, AVBSquare
import torch
import torch.nn as nn
import snntorch as snn
from random import shuffle
import tqdm

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
model.retest_all_classifiers(te_ds_a, te_ds_v)
# model.train_classifiers(tr_ds_a, tr_ds_v, te_ds_a, te_ds_v, epochs=5, lr=1e-3, device=device)

# torch.save(model.state_dict(), f"./model_saves/av_bsquare.pth")
