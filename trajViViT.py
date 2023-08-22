import torch
import torch.nn.functional as F
from torch import nn
import math
from einops import rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_3d(patches, temperature=10000, dtype=torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device=device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6)))  # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TrajViVit(nn.Module):

    def __init__(self, *, dim, depth, heads, mlp_dim, channels, patch_size, nprev, device):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.patch_size = patch_size
        self.nprev = nprev
        self.device = device

        self.conv1 = nn.Conv3d(in_channels=self.channels, out_channels=8, kernel_size=(self.nprev, 3, 3), stride=1,
                               padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AdaptiveAvgPool3d((self.nprev, 64, 64))

        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(self.nprev, 3, 3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool3d((self.nprev, 32, 32))

        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(self.nprev, 3, 3), stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool3d((self.nprev, 16, 16))

        self.pe = PositionalEncoding(self.dim)

        self.encoderLayer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.heads, dim_feedforward=self.mlp_dim,
                                                       batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoderLayer, num_layers=self.depth)
        self.decoderLayer = nn.TransformerDecoderLayer(d_model=self.dim, nhead=self.heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoderLayer, num_layers=self.depth)

        self.coord_to_emb = nn.Linear(2, dim)
        self.emb_to_coord = nn.Linear(dim, 2)
    def forward(self, video, tgt, train=True):
        b, f, h, w, dtype = *video.shape, video.dtype
        video = torch.reshape(video, (b, 1, f, h, w))
        out = self.pool1(self.relu1(self.conv1(video)))
        out = self.pool2(self.relu2(self.conv2(out)))
        x = self.pool3(self.relu3(self.conv3(out)))
        pe = posemb_sincos_3d(x)#posemb_sincos_3d(xx)
        x = rearrange(x, 'b ... d -> b (...) d') #+ self.pe(x)#pe

        x += self.pe(x)
        x = self.encoder(x)

        x = self.generate_sequence(tgt, x, train)
        output = x

        x = self.emb_to_coord(x[:, :-1, :])

        if train:
            return x
        else:
            return x, output

    def generate_sequence(self, tgt, memory, train):
        # Initialize the decoder input with a special start-of-sequence token

        if tgt is not None:

            if train:
                tgt = self.coord_to_emb(tgt)

            sos = torch.ones(memory.shape[0], 1, self.dim).to(self.device)
            tgt = torch.cat([sos, tgt], dim=1)
        else:
            tgt = torch.ones(memory.shape[0], 2, self.dim).to(self.device)

        mask = torch.ones((tgt.shape[0] * self.heads, tgt.shape[1], tgt.shape[1])).to(self.device)
        mask = mask.masked_fill(torch.tril(torch.ones((tgt.shape[1], tgt.shape[1])).to(self.device)) == 0,
                                float('-inf'))
        tgt = self.pe(tgt)
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)

        return output
