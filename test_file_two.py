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
data_dir = home + "/data/av_shd"

# Retrieve cached dataset saved at data_dir
av_ds = SavedAVDataset(data_dir)
print(av_ds[0][0][0].shape, av_ds[0][0][1].shape, av_ds[0][1])

model = AVBSquare()
print(model)

av_ds.shuffle_data()
tr_ub = int(0.8 * len(av_ds))
tr_ds, te_ds = torch.utils.data.random_split(av_ds, [tr_ub, len(av_ds) - tr_ub])
tr_ds_a = tr_ds.dataset.get_a_ds()
te_ds_a = te_ds.dataset.get_a_ds()
tr_ds_v = tr_ds.dataset.get_v_ds()
te_ds_v = te_ds.dataset.get_v_ds()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.train_classifiers(tr_ds_a, te_ds_a, tr_ds_v, te_ds_v, epochs=5, lr=1e-3)

torch.save(model.state_dict(), f"./model_saves/av_bsquare.pth")
