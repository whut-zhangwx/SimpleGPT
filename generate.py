import os
import pickle

import torch

from gpt import GPT
from config import GPTConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_path = "./out/ckpt_iter_50000.pt"

checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
model_args = dict()
for key in ['n_layer', 'n_head', 'embed_dim', 'time_step', 'bias', 'vocab_size']:
    model_args[key] = checkpoint_model_args[key]
# create the model
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

state_dict = checkpoint['state_dict']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
# The bug is when you use torch.compile(model) to compile your model,
# your model.state_dict() will got this "_orig_mod." prefix.
# If you did not use torch.compile(), you could ignore these codes.
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)

print(model)

# load vocab_size, stoi, itos from meta.pkl
# https://docs.python.org/3/library/pickle.html
meta_path = "./tinyshakespeare/meta.pkl"
assert os.path.exists(meta_path), f"'{meta_path}' dose not exit, please run prepare.py first"
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
# gptconf.vocab_size = meta['vocab_size']
stoi = meta['stoi']
itos = meta['itos']

# define encoder and decoder
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# generate from the model
model.eval()
sentence = "First Citizen:\nBefore we proceed any further, hear me speak."
print(f"input:\n{sentence}")
print("---------------")
print("start  generate")
print("---------------")
encoded_sentence = encode(sentence)
input = torch.tensor(encoded_sentence, dtype=torch.long, device=device)
input = input.unsqueeze(0) # (batch_size, time_step)
print(decode(model.generate(input, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
print("---------------")
print("finish generate")
