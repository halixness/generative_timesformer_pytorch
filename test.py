import torch
from timesformer_pytorch import TimeSformer

model = TimeSformer(
    dim = 32**2,
    image_size = 224,
    patch_size = 16,
    num_frames = 8,
    num_target_frames = 4,
    depth = 12,
    heads = 8,
    dim_head =  64,
    attn_dropout = 0.1,
    ff_dropout = 0.1,
    
)

video = torch.randn(2, 8, 3, 224, 224) # (batch x frames x channels x height x width)
pred = model(video) # (2, 4, 3, 224, 224)