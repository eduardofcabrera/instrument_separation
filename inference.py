import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from torch.utils.data import DataLoader
from dataset import MusDBTrackDataset
from model import Model, Model_GPT, Model_Split
from transformers import GPT2Config
from tqdm import tqdm

from pathlib import Path

import random

random.seed(42)

def inference(music_path, model, device):
    
    batch_size = 64
    inference_dataset = MusDBTrackDataset('musdb18hq/train', 8000, 60*1e-3, str(music_path))
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    bass_list = []
    drums_list = []
    vocals_list = []
    bass_hat_list = []
    drums_hat_list = []
    vocals_hat_list = []
    for batch in tqdm(inference_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        mixture = batch['mixture']
        mixture_freq = batch['mixture_freq']

        bass_hat, drums_hat, vocals_hat = model(mixture, mixture_freq)
        
        bass_list.append(batch['bass'].squeeze().detach().cpu())
        drums_list.append(batch['drums'].squeeze().detach().cpu())
        vocals_list.append(batch['vocals'].squeeze().detach().cpu())    
        
        bass_hat_list.append(bass_hat.detach().cpu())
        drums_hat_list.append(drums_hat.detach().cpu())
        vocals_hat_list.append(vocals_hat.detach().cpu())

    bass = torch.cat(bass_list, axis=0).reshape(-1).numpy()
    drums = torch.cat(drums_list, axis=0).reshape(-1).numpy()
    vocals = torch.cat(vocals_list, axis=0).reshape(-1).numpy()
    
    bass_hat = torch.cat(bass_hat_list, axis=0).reshape(-1).numpy()
    drums_hat = torch.cat(drums_hat_list, axis=0).reshape(-1).numpy()
    vocals_hat = torch.cat(vocals_hat_list, axis=0).reshape(-1).numpy()

    df = pd.DataFrame({
        "bass": bass,
        "bass_hat": bass_hat,
        "drums": drums,
        "drums_hat": drums_hat,
        "vocals": vocals,
        "vocals_hat": vocals_hat,
    })
    
    return df 

def run_inferences(train: bool = True, model: str = "2_loss"):
    
    if model == "2_loss":
        model_path = "cnn_model_2_loss/model.pt"
    elif model == "index":
        model_path = "cnn_model_index/model.pt"
    elif model == "freq":
        model_path = "cnn_freq_loss/model.pt"
    elif model == "split":
        model_path = "cnn_split/model.pt"
    
    dataset_path = Path('musdb18hq/train') if train else Path('musdb18hq/val')
    inference_folder = Path('inference') / model / "train" if train else Path('inference') / model / "val"
    inference_folder.mkdir(parents=True, exist_ok=True)
    musics = list(dataset_path.glob('*'))
    
    if train:
        musics = random.sample(musics, 10)
    
    #model = Model(3, 2000, 4)
    model = Model_Split(1, 1000, 2)
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for i, music in enumerate(musics):
        print(f'Inference {i+1}/{len(musics)}')
        df = inference(music, model, device)
        df_file = inference_folder / f'{music.stem}.csv'
        df.to_csv(df_file, index=False)
    



if __name__ == '__main__':
    run_inferences(train=False, model="split")