import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import math

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        num_tokens = 1,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.num_tokens = num_tokens

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q *= self.scale

        # splitting out added target tokens
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:self.num_tokens], t[:, self.num_tokens:]), (q, k, v))

        # target tokens attend all the tokens in sequence
        cls_out = attn(cls_q, k, v)

        # rearrange across time or space
        # 'b (f n) d' -> '(b n) f d'
        # 'b (f n) d' -> '(b f) n d'
        # time = batch, frame * num token, dim -> batch * num patch, frame, dim -> iters frame
        # space = batch, frame * num token, dim -> batch * num frame, patch, dim -> iters patch
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]

        cls_k, cls_v = map(lambda t: repeat(t, 'b t d -> (b r) t d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim = 1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

# main classes

class TimeSformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_target_frames = 4,
        image_size = 224,
        patch_size = 16,
        channels = 3,
        out_channels = 3,
        depth = 12,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.dim = dim
        self.out_channels = out_channels

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2
        out_patch_dim = out_channels * patch_size ** 2

        self.num_tokens = num_target_frames
        self.num_target_patches = self.num_tokens * num_patches

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        # extend positional embedding, define the target tokens
        self.pos_emb = nn.Embedding(num_positions + self.num_target_patches, dim)
        self.target_tokens = nn.Parameter(torch.randn(self.num_target_patches, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):

            # time_attn, space_attn, ff
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, num_tokens = self.num_target_patches)),
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, num_tokens = self.num_target_patches)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout))
            ]))

        self.to_dembedded_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_patch_dim)
        )

    def forward(self, video):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        n = (h // p) * (w // p)
        
        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = p, p2 = p)

        # from patch size to embedding size
        tokens = self.to_patch_embedding(video)

        target_tokens = repeat(self.target_tokens, 'n d -> b n d', b = b)
        x =  torch.cat((target_tokens, tokens), dim = 1) 
        x += self.pos_emb(torch.arange(x.shape[1], device = device))

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n = n) + x # 1 patch over N frames
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f) + x # N patches for 1 frame
            x = ff(x) + x
            break

        # out_tokens <- num_target_frames * num_patches
        out_tokens = x[:, 0:self.num_target_patches]

        # embed dim -> original patch dim
        out_tokens = self.to_dembedded_out(out_tokens)

        return out_tokens.view(b, self.num_tokens, self.out_channels, h, w)
