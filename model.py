import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
import random

random.seed(42)

class RNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        #self.norm_1 = nn.BatchNorm1d(in_channels)
        self.rnn = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.norm_2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        #x_ = x.swapaxes(1, 2)
        #x_ = self.norm_1(x_)
        #x_ = x_.swapaxes(1, 2)
        x_ = self.rnn(x)[0]
        x_ = x_.swapaxes(1, 2)
        x_ = self.norm_2(x_)
        x_ = x_.swapaxes(1, 2)
        return x_

class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.gru_time_domain = RNNEncoder(in_channels=in_channels, hidden_size=hidden_size, num_layers=num_layers)
        self.gru_freq_domain = RNNEncoder(in_channels=in_channels, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x_time_domain, x_freq_domain):
        time_domain = self.gru_time_domain(x_time_domain)
        freq_domain = self.gru_freq_domain(x_freq_domain)

        return time_domain, freq_domain

class RNNDecoder(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers):
        super(RNNDecoder, self).__init__()
        self.norm_1 = nn.BatchNorm1d(hidden_size)
        self.rnn = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, x, timestamp):
        x = x.swapaxes(1, 2)
        x_ = self.norm_1(x)
        x_ = x_.swapaxes(1, 2)

        #x_ = self.rnn(x_)[0]
        #x_ = self.projection(x_)
        outputs = torch.empty((x.shape[0], timestamp.shape[-1], 1), dtype=torch.float32, device=x.device)

        forecast = 0
        #rnn_input = self.projection(x_[:, 0, :]) / 2
        #outputs[:, 0], _ = rnn_input
        for i in range(0, timestamp.shape[-1]):
            rnn_input = (forecast + self.projection(x_[:, i, :]))/2
            rnn_input = rnn_input.unsqueeze(1)
            output, _ = self.rnn(rnn_input)
            forecast = self.projection(output.squeeze())
            outputs[:, i] = forecast
            #rnn_input = (forecast + self.projection(x_[:, i, :]))/2

        return outputs


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.bass_decoder = RNNDecoder(in_channels=1, hidden_size=hidden_size, num_layers=num_layers)
        self.drumns_decoder = RNNDecoder(in_channels=1, hidden_size=hidden_size, num_layers=num_layers)
        self.vocals_decoder = RNNDecoder(in_channels=1, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x_bass, x_drums, x_vocals, timestamp):
        bass = self.bass_decoder(x_bass, timestamp)
        drums = self.drumns_decoder(x_drums, timestamp)
        vocals = self.vocals_decoder(x_vocals, timestamp)
        return bass, drums, vocals

class Agregator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Agregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.mlp(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], x.shape[1], x.shape[2]//2)

class Model(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers):
        super(Model, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = Decoder(in_channels=hidden_size, hidden_size=hidden_size, num_layers=1)

        self.agregator_bass = Agregator(in_channels=2*hidden_size, out_channels=hidden_size)
        self.agregator_drums = Agregator(in_channels=2*hidden_size, out_channels=hidden_size)
        self.agregator_vocals = Agregator(in_channels=2*hidden_size, out_channels=hidden_size)

        self.freq_linear_bass = nn.Linear(120, 480)
        self.freq_linear_drums = nn.Linear(120, 480)
        self.freq_linear_vocals = nn.Linear(120, 480)

    def forward(self, x_time_domain, x_freq_domain):
        timestamp = x_time_domain[:, :, -1]
        time_domain, freq_domain = self.encoder(x_time_domain, x_freq_domain)
        freq_domain = freq_domain.swapaxes(1, 2)
        freq_domain = self.freq_linear_bass(freq_domain)
        freq_domain = freq_domain.swapaxes(1, 2)

        hidden_bass = self.agregator_bass(torch.concat([time_domain, freq_domain], axis=-1))
        hidden_drums = self.agregator_drums(torch.concat([time_domain, freq_domain], axis=-1))
        hidden_vocals = self.agregator_vocals(torch.concat([time_domain, freq_domain], axis=-1))

        bass, drums, vocals = self.decoder(hidden_bass, hidden_drums, hidden_vocals, timestamp)
        return bass, drums, vocals
    
class Model_Split(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers):
        super(Model_Split, self).__init__()
        self.encoder_bass = Encoder(in_channels=in_channels, hidden_size=hidden_size, num_layers=num_layers)
        self.encoder_drums = Encoder(in_channels=in_channels, hidden_size=hidden_size, num_layers=num_layers)
        self.encoder_vocals = Encoder(in_channels=in_channels, hidden_size=hidden_size, num_layers=num_layers)
        
        self.decoder = Decoder(in_channels=hidden_size, hidden_size=hidden_size, num_layers=1)

        self.agregator_bass = Agregator(in_channels=2*hidden_size, out_channels=hidden_size)
        self.agregator_drums = Agregator(in_channels=2*hidden_size, out_channels=hidden_size)
        self.agregator_vocals = Agregator(in_channels=2*hidden_size, out_channels=hidden_size)

        self.freq_linear_bass = nn.Linear(120, 480)
        self.freq_linear_drums = nn.Linear(120, 480)
        self.freq_linear_vocals = nn.Linear(120, 480)

    def forward(self, x_time_domain, x_freq_domain):
        
        timestamp = torch.arange(x_time_domain.shape[-1])
        
        x_time_domain = x_time_domain.unsqueeze(-1)
        x_freq_domain = x_freq_domain.unsqueeze(-1)
        time_domain_bass, freq_domain_bass = self.encoder_bass(x_time_domain, x_freq_domain)
        freq_domain_bass = freq_domain_bass.swapaxes(1, 2)
        freq_domain_bass = self.freq_linear_bass(freq_domain_bass)
        freq_domain_bass = freq_domain_bass.swapaxes(1, 2)
        
        time_domain_drums, freq_domain_drums = self.encoder_drums(x_time_domain, x_freq_domain)
        freq_domain_drums = freq_domain_drums.swapaxes(1, 2)
        freq_domain_drums = self.freq_linear_drums(freq_domain_drums)
        freq_domain_drums = freq_domain_drums.swapaxes(1, 2)
        
        time_domain_vocals, freq_domain_vocals = self.encoder_vocals(x_time_domain, x_freq_domain)
        freq_domain_vocals = freq_domain_vocals.swapaxes(1, 2)
        freq_domain_vocals = self.freq_linear_vocals(freq_domain_vocals)
        freq_domain_vocals = freq_domain_vocals.swapaxes(1, 2)

        hidden_bass = self.agregator_bass(torch.concat([time_domain_bass, freq_domain_bass], axis=-1))
        hidden_drums = self.agregator_drums(torch.concat([time_domain_drums, freq_domain_drums], axis=-1))
        hidden_vocals = self.agregator_vocals(torch.concat([time_domain_vocals, freq_domain_vocals], axis=-1))

        bass, drums, vocals = self.decoder(hidden_bass, hidden_drums, hidden_vocals, timestamp)
        return bass, drums, vocals
    

class GPT2_TimeSeries_Forecasting(nn.Module):
    def __init__(self, config, n_dims_in, n_dims_out):
        super().__init__()
        self.read_in = nn.Linear(n_dims_in, config.n_embd)
        self.gpt2 = GPT2Model(config)
        self.read_out = nn.Linear(config.n_embd, n_dims_out)
    
    def forward(self, x):
        embeds = self.read_in(x)
        output = self.gpt2(inputs_embeds=embeds).last_hidden_state
        prediction = self.read_out(output)
        return prediction
    
class Model_GPT(nn.Module):
    def __init__(self, config, n_dims_in, n_dims_out):
        super().__init__()
        self.bass = GPT2_TimeSeries_Forecasting(config, n_dims_in, n_dims_out)
        self.drums = GPT2_TimeSeries_Forecasting(config, n_dims_in, n_dims_out)
        self.vocals = GPT2_TimeSeries_Forecasting(config, n_dims_in, n_dims_out)
        
    def forward(self, x):
        bass_hat = self.bass(x)
        drums_hat = self.drums(x)
        vocals_hat = self.vocals(x)
        
        return bass_hat, drums_hat, vocals_hat