from pathlib import Path
from typing import *
import numpy as np
import random
import torchaudio
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import librosa

random.seed(42)


class MusDBDatasetInDistribution(Dataset):
    def __init__(self, base_folder: str, sample_rate: int, n_tracks: int, window_size: float, train: bool):
        self.base_folder = Path(base_folder)
        self.sample_rate = sample_rate
        self.n_tracks = n_tracks
        self.window_size = window_size
        self.data = self.get_data()
        self.len = self.__len__()
        self.train = train
        
    def load_wav(self, path: Path) -> torch.Tensor:
        data, sample_rate = torchaudio.load(str(path))
        data = torchaudio.functional.resample(data, sample_rate, self.sample_rate)
        return data
    
    def get_data(self):
        base_folder = self.base_folder
        tracks = list(base_folder.glob('*'))
        if self.n_tracks != -1:
            tracks = random.sample(tracks, self.n_tracks)
        print(tracks)
        mixture_frames_concat = torch.tensor([])
        bass_frames_concat = torch.tensor([])
        drums_frames_concat = torch.tensor([])
        vocals_frames_concat = torch.tensor([])
        for track_folder in tqdm(tracks):
            mixture = track_folder / 'mixture.wav'
            bass = track_folder / 'bass.wav'
            drums = track_folder / 'drums.wav'
            vocals = track_folder / 'vocals.wav'
            other = track_folder / 'other.wav'

            mixture_data = self.load_wav(mixture).mean(axis=0)
            bass_data = self.load_wav(bass).mean(axis=0)
            drums_data = self.load_wav(drums).mean(axis=0)
            vocals_data = self.load_wav(vocals).mean(axis=0)
            
            frame_length = int(self.window_size*self.sample_rate)
            mixture_frames = torch.tensor(librosa.util.frame(mixture_data, frame_length=frame_length, hop_length=frame_length))
            bass_frames = torch.tensor(librosa.util.frame(bass_data, frame_length=frame_length, hop_length=frame_length))
            drums_frames = torch.tensor(librosa.util.frame(drums_data, frame_length=frame_length, hop_length=frame_length))
            vocals_frames = torch.tensor(librosa.util.frame(vocals_data, frame_length=frame_length, hop_length=frame_length))
            
            mixture_frames_concat = torch.concat([mixture_frames_concat, mixture_frames], axis=-1)
            bass_frames_concat = torch.concat([bass_frames_concat, bass_frames], axis=-1)
            drums_frames_concat = torch.concat([drums_frames_concat, drums_frames], axis=-1)
            vocals_frames_concat = torch.concat([vocals_frames_concat, vocals_frames], axis=-1)
            
        return {
            "mixture": mixture_frames_concat,
            "bass": bass_frames_concat,
            "drums": drums_frames_concat,
            "vocals": vocals_frames_concat,
        }
        
    def __len__(self):

        return int(self.data["mixture"].shape[-1]/2)
    
    def __getitem__(self, index) -> Any:

        index = int(index*2)
        if not self.train:
            index += 1
        
        mixture_frame = self.data["mixture"][:, index]
        bass_frame = self.data["bass"][:, index]
        drums_frame = self.data["drums"][:, index]
        vocals_frame = self.data["vocals"][:, index]
        
        mixture_fft = torch.fft.rfft(mixture_frame)
        mag = torch.abs(mixture_fft[0:int((mixture_fft.shape[-1]-1)/2)])

        return {
            "mixture": mixture_frame,
            "mixture_freq": mag,
            "bass": bass_frame,
            "drums": drums_frame,
            "vocals": vocals_frame
        }
                
class MusDBDataset(Dataset):
    def __init__(self, base_folder: str, sample_rate: int, n_tracks: int, window_size: float):
        self.base_folder = Path(base_folder)
        self.sample_rate = sample_rate
        self.n_tracks = n_tracks
        self.window_size = window_size
        self.data = self.get_data()
        self.len = self.__len__()

    def load_wav(self, path: Path) -> torch.Tensor:
        data, sample_rate = torchaudio.load(str(path))
        data = torchaudio.functional.resample(data, sample_rate, self.sample_rate)
        return data

    def get_data(self):
        base_folder = self.base_folder
        tracks = list(base_folder.glob('*'))
        if self.n_tracks != -1:
            tracks = random.sample(tracks, self.n_tracks)
        print(tracks)
        data = []
        for track_folder in tqdm(tracks):
            mixture = track_folder / 'mixture.wav'
            bass = track_folder / 'bass.wav'
            drums = track_folder / 'drums.wav'
            vocals = track_folder / 'vocals.wav'
            other = track_folder / 'other.wav'

            mixture_data = self.load_wav(mixture)
            bass_data = self.load_wav(bass)
            drums_data = self.load_wav(drums)
            vocals_data = self.load_wav(vocals)
            timestamp = torch.arange(0,mixture_data.shape[-1])/self.sample_rate

            data.append({
                "mixture": mixture_data,
                "bass": bass_data,
                "drums": drums_data,
                "vocals": vocals_data,
                "timestamp": timestamp,
            })

        return data

    def __len__(self):

        count = 0
        for track in self.data:
            timestamp = track['timestamp']
            n_time = timestamp[-1]
            n_windows = n_time // self.window_size
            count += n_windows
        return int(count)

    def __getitem__(self, index) -> Any:

        track = random.choice(self.data)
        timestamp = track['timestamp']
        context_window_size = int(self.window_size*self.sample_rate)

        start_point = random.randint(0, timestamp.shape[0] - context_window_size)
        end_point = start_point + context_window_size

        timestamp = timestamp[start_point:end_point]
        mixture = track['mixture'][:, start_point:end_point]
        bass = track['bass'][:, start_point:end_point].mean(axis=0).unsqueeze(0)
        drums = track['drums'][:, start_point:end_point].mean(axis=0).unsqueeze(0)
        vocals = track['vocals'][:, start_point:end_point].mean(axis=0).unsqueeze(0)

        mixture_fft = torch.fft.rfft(mixture)
        mag = torch.abs(mixture_fft[:, 0:int((mixture_fft.shape[-1]-1)/2)])
        freqs = torch.arange(0, int((mixture_fft.shape[-1]-1)/2))*(1/self.window_size)

        mixture = torch.concatenate([mixture, timestamp.unsqueeze(0)], axis=0)
        mag = torch.concatenate([mag, freqs.unsqueeze(0)], axis=0)

        return {
            "mixture": mixture.T,
            "mixture_freq": mag.T,
            "bass": bass.T,
            "drums": drums.T,
            "vocals": vocals.T,
        }

class MusDBTrackDataset(MusDBDataset):
    def __init__(self, base_folder: str, sample_rate: int, window_size: float, track_folder: str):
        self.base_folder = Path(base_folder)
        self.sample_rate = sample_rate
        self.n_tracks = 1
        self.window_size = window_size
        self.data = self.get_data(track_folder)
        self.len = self.__len__()

    def load_wav(self, path: Path) -> torch.Tensor:
        data, sample_rate = torchaudio.load(str(path))
        data = torchaudio.functional.resample(data, sample_rate, self.sample_rate)
        return data

    def get_data(self, music_folder: str):
        base_folder = self.base_folder
        track_folder = Path(music_folder)
        #tracks = random.sample(tracks, self.n_tracks)
        print(track_folder)
        data = []
        mixture = track_folder / 'mixture.wav'
        bass = track_folder / 'bass.wav'
        drums = track_folder / 'drums.wav'
        vocals = track_folder / 'vocals.wav'
        other = track_folder / 'other.wav'

        mixture_data = self.load_wav(mixture)
        bass_data = self.load_wav(bass)
        drums_data = self.load_wav(drums)
        vocals_data = self.load_wav(vocals)
        timestamp = torch.arange(0,mixture_data.shape[-1])/self.sample_rate

        data.append({
            "mixture": mixture_data,
            "bass": bass_data,
            "drums": drums_data,
            "vocals": vocals_data,
            "timestamp": timestamp,
        })

        return data

    def __len__(self):

        count = 0
        for track in self.data:
            timestamp = track['timestamp']
            n_time = timestamp[-1]
            n_windows = n_time // self.window_size
            count += n_windows
        return int(count)

    def __getitem__(self, index) -> Any:

            track = self.data[0]
            timestamp = track['timestamp']
            context_window_size = int(self.window_size*self.sample_rate)

            start_point = index*context_window_size
            end_point = start_point + context_window_size

            timestamp = timestamp[start_point:end_point]
            mixture = track['mixture'][:, start_point:end_point].mean(axis=0)
            bass = track['bass'][:, start_point:end_point].mean(axis=0)
            drums = track['drums'][:, start_point:end_point].mean(axis=0)
            vocals = track['vocals'][:, start_point:end_point].mean(axis=0)

            mixture_fft = torch.fft.rfft(mixture)
            mag = torch.abs(mixture_fft[0:int((mixture_fft.shape[-1]-1)/2)])
            #freqs = torch.arange(0, int((mixture_fft.shape[-1]-1)/2))*(1/self.window_size)

            #mixture = torch.concatenate([mixture, timestamp.unsqueeze(0)], axis=0)
            #mag = torch.concatenate([mag, freqs.unsqueeze(0)], axis=0)

            return {
                "mixture": mixture,
                "mixture_freq": mag,
                "bass": bass,
                "drums": drums,
                "vocals": vocals,
            }


if __name__ == '__main__':
    dataset = MusDBDatasetInDistribution('musdb18hq/train', 8000, 1, 60*1e-3, train=True)
    dataset[0]