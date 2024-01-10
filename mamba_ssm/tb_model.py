import torch
import torch.nn.functional as F
from einops import rearrange

print(torch.cuda.is_available())












# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("state-spaces/mamba-130m")