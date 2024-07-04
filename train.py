import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import MusDBDataset, MusDBDatasetInDistribution
from model import Model, Model_GPT, Model_Split
from tqdm import tqdm
from torchmetrics.audio import SignalDistortionRatio
import random
from copy import deepcopy

from transformers import GPT2Config

import uniplot

def index_of_agreement(s, o):
    ia = 1 - (torch.sum((o - s) ** 2, dim=0)) / (
        torch.sum(
            (torch.abs(s - torch.mean(o, dim=0)) + torch.abs(o - torch.mean(o, dim=0)))
            ** 2,
            dim=0,
        )
    )

    return ia.mean()

IndexOfAgreementLoss = lambda: lambda s, o: -index_of_agreement(s, o) + 1.0

def loss_func_with_freq(y_hat, y):
    #index_of_agreement_loss = IndexOfAgreementLoss()
    #index_of_agreement = index_of_agreement_loss(y_hat, y)

    fft_hat = torch.fft.rfft(y_hat, axis=1)
    mag_hat = torch.abs(fft_hat[:, 0:int((fft_hat.shape[1]-1)/2)])

    fft = torch.fft.rfft(y, axis=1)
    mag = torch.abs(fft[:, 0:int((fft.shape[1]-1)/2)])

    mse = nn.MSELoss()
    freq_loss = mse(mag_hat, mag)

    return index_of_agreement#freq_loss #index_of_agreement #+ freq_loss


LossFunc = lambda: lambda y_hat, y: loss_func_with_freq(y_hat, y)

random.seed(42)

def train():

    configuration = GPT2Config(
            n_positions=2048, # set to sthg large advised
            n_embd=256,
            n_layer=16,
            n_head=16,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
    )

    epochs = 100
    lr = 1e-4
    batch_size = 64
    val_batch_sise = 32
    #model = Model_GPT(configuration, 1, 1)
    #train_dataset = MusDBDataset('musdb18hq/train', 16000, -1, 30*1e-3)
    train_dataset = MusDBDatasetInDistribution('musdb18hq/train', 8000, 30, 60*1e-3, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #val_dataset = MusDBDataset('musdb18hq/val', 16000, -1, 30*1e-3)
    val_dataset = deepcopy(train_dataset)
    val_dataset.train = False
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_sise, shuffle=True)

    #model = Model(3, 2000, 4)
    model = Model_Split(1, 1000, 2)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of Parameters: {pytorch_total_params // 1e6}M")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #device_1 = "cuda:0"
    #device_2 = "cuda:1"
    #model.load_state_dict(torch.load('cnn_model_2_loss/model-0_9599.pt'))
    model.to(device)
    
    #model.bass.to(device_1)
    #model.drums.to(device_1)
    #model.vocals.to(device_2)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    #loss_func = nn.MSELoss()
    loss_func = IndexOfAgreementLoss()
    #loss_func = LossFunc()
    #loss_func = SignalDistortionRatio()

    batch_n = 50

    for epoch in range(epochs):
        bass_loss_list = []
        drums_loss_list = []
        vocals_loss_list = []
        loss_list = []
        len_batch = len(train_dataloader)
        pbar = tqdm(total=batch_n)
        for n_batch, batch in enumerate(train_dataloader):
            pbar.update(1)
            model.train()
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            mixture = batch['mixture']
            #mixture = mixture[:, :, :2].mean(axis=-1).unsqueeze(-1)
            
            mixture_freq = batch['mixture_freq']
            bass = batch['bass']
            drums = batch['drums']
            vocals = batch['vocals']

            bass_hat, drums_hat, vocals_hat = model(mixture, mixture_freq)
            bass_hat = bass_hat.squeeze()
            drums_hat = drums_hat.squeeze()
            vocals_hat = vocals_hat.squeeze()
            #bass_hat, drums_hat, vocals_hat = model(mixture)

            bass_loss = loss_func(bass_hat, bass)
            drums_loss = loss_func(drums_hat, drums)
            vocals_loss = loss_func(vocals_hat, vocals)
            loss = bass_loss + drums_loss + vocals_loss

            bass_loss_list.append(bass_loss.item())
            drums_loss_list.append(drums_loss.item())
            vocals_loss_list.append(vocals_loss.item())
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            if (n_batch+1) % batch_n == 0:
                pbar.reset()
                mean_bass_loss = sum(bass_loss_list) / len(bass_loss_list)
                bass_loss_list = []
                mean_drums_loss = sum(drums_loss_list) / len(drums_loss_list)
                drums_loss_list = []
                mean_vocals_loss = sum(vocals_loss_list) / len(vocals_loss_list)
                vocals_loss_list = []
                mean_loss = sum(loss_list) / len(loss_list)
                loss_list = []
                print(f'Batch: {n_batch}/{len_batch}\nTrain\nBass Loss: {mean_bass_loss}\nDrums Loss: {mean_drums_loss}\nVocals Loss: {mean_vocals_loss}\nTotal Loss: {mean_loss}')

                model.eval()
                val_batch = next(iter(val_dataloader))
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                mixture = val_batch['mixture']
                #mixture = mixture[:, :, :2].mean(axis=-1).unsqueeze(-1)
                mixture_freq = val_batch['mixture_freq']
                bass = val_batch['bass']
                drums = val_batch['drums']
                vocals = val_batch['vocals']

                bass_hat, drums_hat, vocals_hat = model(mixture, mixture_freq)
                bass_hat = bass_hat.squeeze()
                drums_hat = drums_hat.squeeze()
                vocals_hat = vocals_hat.squeeze()
                
                bass_loss = loss_func(bass_hat, bass)
                drums_loss = loss_func(drums_hat, drums)
                vocals_loss = loss_func(vocals_hat, vocals)
                loss = bass_loss + drums_loss + vocals_loss

                bass_example = bass[0].detach().cpu().numpy()
                bass_example_hat = bass_hat[0].detach().cpu().numpy()

                drums_example = drums[0].detach().cpu().numpy()
                drums_example_hat = drums_hat[0].detach().cpu().numpy()

                vocals_example = vocals[0].detach().cpu().numpy()
                vocals_example_hat = vocals_hat[0].detach().cpu().numpy()

                uniplot.plot([bass_example, bass_example_hat], title='Bass', color=True, legend_labels=["y", "y_hat"])
                uniplot.plot([drums_example, drums_example_hat], title='Drums', color=True, legend_labels=["y", "y_hat"])
                uniplot.plot([vocals_example, vocals_example_hat], title='Vocals', color=True, legend_labels=["y", "y_hat"])
                print(f'Validation\nBass Loss: {bass_loss.item()}\nDrums Loss: {drums_loss.item()}\nVocals Loss: {vocals_loss.item()}\nTotal Loss: {loss.item()}')
                torch.save(model.state_dict(), f'cnn_split/model_20.pt')

        #mean_bass_loss = sum(bass_loss_list) / len(bass_loss_list)
        #mean_drums_loss = sum(drums_loss_list) / len(drums_loss_list)
        #mean_vocals_loss = sum(vocals_loss_list) / len(vocals_loss_list)
        #mean_loss = sum(loss_list) / len(loss_list)

        #print(f'Epoch {epoch+1}/{epochs} - Bass Loss: {mean_bass_loss} - Drums Loss: {mean_drums_loss} - Vocals Loss: {mean_vocals_loss} - Total Loss: {mean_loss}')


if __name__ == "__main__":
    train()