import torch
import os
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b-slimpj", device="cuda", dtype=torch.bfloat16)
tokens = tokenizer("Once upon a time, a cat named", return_tensors="pt")
input_ids = tokens.input_ids.to(device="cuda")
max_length = input_ids.shape[1] + 80
fn = lambda: model.generate(
        input_ids=input_ids, max_length=max_length, cg=True,
        return_dict_in_generate=True, output_scores=True,
        enable_timing=False, temperature=0.9, top_k=40, top_p=0.9,)
out = fn()
print(tokenizer.decode(out[0][0]))
    









