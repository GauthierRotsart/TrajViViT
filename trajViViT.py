import torch
from torch import nn
import math
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_r = dropout
        self.device = device

        if self.dropout_r > 0:
            self.dropout = nn.Dropout(self.dropout_r)

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe[:x.size(0), :].to(self.device)
        return self.dropout(x)


class TrajViVit(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, channels, dropout, n_prev, pos_bool, img_bool, device):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.dropout_r = dropout
        self.n_prev = n_prev
        self.pos_bool = pos_bool
        self.img_bool = img_bool
        self.device = device
        if self.pos_bool is True and self.img_bool is True:
            self.dim *= 2

        if self.dropout_r > 0:
            self.dropout = nn.Dropout(self.dropout_r)

        self.conv1 = nn.Conv3d(in_channels=self.channels, out_channels=8, kernel_size=(self.n_prev, 3, 3))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AdaptiveAvgPool3d((self.n_prev, 64, 64))

        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(self.n_prev, 3, 3))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool3d((self.n_prev, 32, 32))

        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(self.n_prev, 3, 3))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool3d((self.n_prev, 16, 16))

        self.pe = PositionalEncoding(d_model=self.dim, dropout=self.dropout_r, device=self.device)

        self.encoderLayer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.heads, dim_feedforward=self.mlp_dim,
                                                       batch_first=True, dropout=self.dropout_r)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoderLayer, num_layers=self.depth)
        self.decoderLayer = nn.TransformerDecoderLayer(d_model=self.dim, nhead=self.heads, batch_first=True,
                                                       dropout=self.dropout_r)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoderLayer, num_layers=self.depth)

        self.coord_to_emb = nn.Linear(2, self.dim)
        self.emb_to_coord = nn.Linear(self.dim, 2)
        if self.pos_bool is True and self.img_bool is True:
            self.src_to_emb = nn.Linear(self.n_prev * 2, self.n_prev * 32 * self.dim * self.dim // 4)
        else:
            self.src_to_emb = nn.Linear(self.n_prev * 2, self.n_prev * 32 * self.dim * self.dim)

    def forward(self, video, tgt, src):
        b, f, h, w = video.shape
        video = torch.reshape(video, (b, 1, f, h, w))

        if self.img_bool:
            out = self.pool1(self.relu1(self.conv1(video)))
            out = self.pool2(self.relu2(self.conv2(out)))
            x = self.pool3(self.relu3(self.conv3(out)))
            x_img = rearrange(x, 'b ... d -> b (...) d')

            if self.pos_bool:
                src = src.contiguous().view(b, -1)
                src = self.src_to_emb(src)
                x_src = src.contiguous().view(b, -1, self.dim // 2)
                x = torch.cat((x_img, x_src), 2)  # x_img + x_src
            else:
                x = x_img
        else:
            if self.pos_bool:
                src = src.contiguous().view(b, -1)
                src = self.src_to_emb(src)
                x_src = src.contiguous().view(b, -1, self.dim)
                x = x_src
            else:
                print("The input is at least the positions or the images.")
                raise NotImplementedError

        x = self.encoder(self.pe(x))
        x = self.generate_sequence(tgt, x)
        output = x  # Autoregressive process on the output embedding
        x = self.emb_to_coord(x[:, :-1, :])
        return x, output

    # DECODER'S INPUT CREATION
    def generate_sequence(self, tgt, memory):
        if tgt is not None:
            self.coord_to_emb = nn.Linear(tgt.shape[2], self.dim).to(self.device)
            sos = torch.ones(tgt.shape[0], 1, tgt.shape[2]).to(self.device)
            tgt = torch.cat([sos, tgt], 1)
        else:
            self.coord_to_emb = nn.Linear(2, self.dim).to(self.device)
            tgt = torch.ones(memory.shape[0], 2, 2).to(self.device)

        mask = torch.ones((tgt.shape[0] * self.heads, tgt.shape[1], tgt.shape[1]))
        mask = mask.masked_fill(torch.tril(torch.ones((tgt.shape[1], tgt.shape[1]))) == 0,
                                float('-inf')).to(self.device)

        tgt = self.coord_to_emb(tgt)
        tgt = self.pe(tgt)
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)

        return output
