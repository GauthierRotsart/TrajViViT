import torch
import torch.nn.functional as F
from torch import nn
import math
from einops import rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

"""
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
"""
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
        #self.device = x.get_device()
        x += self.pe[:x.size(0), :].to(x.get_device())
        return self.dropout(x)


class TrajViVit(nn.Module):

    def __init__(self, *, dim, depth, heads, mlp_dim, channels, patch_size, nprev, device, pos_bool, img_bool, dropout,
                 app=False):
        super().__init__()

        self.app = app
        if self.app:
            self.dim = dim*2
        else:
            self.dim = dim
        self.dim_old = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.patch_size = patch_size
        self.nprev = nprev
        self.device = device
        self.pos_bool = pos_bool
        self.img_bool = img_bool
        self.dropout_r = dropout

        if self.dropout_r > 0:
            self.dropout = nn.Dropout(self.dropout_r)

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
                                                       batch_first=True, dropout=self.dropout_r)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoderLayer, num_layers=self.depth)
        self.decoderLayer = nn.TransformerDecoderLayer(d_model=self.dim, nhead=self.heads, batch_first=True,
                                                       dropout=self.dropout_r)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoderLayer, num_layers=self.depth)

        self.coord_to_emb = nn.Linear(2, self.dim)
        self.emb_to_coord = nn.Linear(self.dim, 2)
        self.src_to_emb = nn.Linear(self.nprev * 2, self.nprev * 32 * self.dim_old * self.dim_old)
    def forward(self, video, tgt, src, train=True):
        b, f, h, w, dtype = *video.shape, video.dtype
        video = torch.reshape(video, (b, 1, f, h, w))
        if self.img_bool:
            out = self.pool1(self.relu1(self.conv1(video)))
            out = self.pool2(self.relu2(self.conv2(out)))
            x = self.pool3(self.relu3(self.conv3(out)))
            #pe = posemb_sincos_3d(x)#posemb_sincos_3d(xx)
            x_img = rearrange(x, 'b ... d -> b (...) d') #+ self.pe(x)#pe

        if self.pos_bool:
            src = src.contiguous().view(b, -1)
            if self.dropout_r > 0:
                src = self.dropout(self.src_to_emb(src))
                x_src = src.contiguous().view(b, -1, self.dim_old)
            else:
                src = self.src_to_emb(src)
                x_src = src.contiguous().view(b, -1, self.dim_old)

        if self.img_bool == True and self.pos_bool == True:
            if self.app:
                x = torch.cat((x_img,x_src),2)#x_img + x_src
            else:
                x = x_img + x_src
        elif self.img_bool == True and self.pos_bool == False:
            x = x_img
        elif self.img_bool == False and self.pos_bool == True:
            x = x_src
        else:
            raise NotImplementedError

        x = self.pe(x)
        x = self.encoder(x)

        x = self.generate_sequence(tgt, x, train)
        output = x#.to(self.device)

        x = self.emb_to_coord(x[:, :-1, :])#.to(self.device)

        if train:
            return x
        else:
            return x, output

    def generate_sequence(self, tgt, memory, train):
        # Initialize the decoder input with a special start-of-sequence token
        if tgt is not None:
            #self.device = tgt.get_device()
            if train:
                tgt = self.coord_to_emb(tgt)
            #tgt = self.coord_to_emb(tgt)
            sos = torch.ones(memory.shape[0], 1, self.dim).to(memory.get_device())
            tgt = torch.cat([sos, tgt], dim=1)
        else:
            tgt = torch.ones(memory.shape[0], 2, self.dim).to(memory.get_device())
        #self.device = tgt.get_device()
        #memory = memory.to(self.device)
        mask = torch.ones((tgt.shape[0] * self.heads, tgt.shape[1], tgt.shape[1]))#.to(self.device)
        #mask = mask.masked_fill(torch.tril(torch.ones((tgt.shape[1], tgt.shape[1])).to(self.device)) == 0,
                               # float('-inf'))
        mask = mask.masked_fill(torch.tril(torch.ones((tgt.shape[1], tgt.shape[1]))) == 0,
                                float('-inf')).to(tgt.get_device())
        tgt = self.pe(tgt)
        #print(tgt.get_device(), memory.get_device(), mask.get_device())
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)

        return output
