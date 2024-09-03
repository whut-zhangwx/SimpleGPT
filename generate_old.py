import os
import pickle

import torch

from gpt_old import GPT
from config import GPTConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path2state_dict = "./state_dict/gpt.pth"
gptconf = GPTConfig()

torch.manual_seed(1337)

# load vocab_size, stoi, itos from meta.pkl
# https://docs.python.org/3/library/pickle.html
meta_path = "./tinyshakespeare/meta.pkl"
assert os.path.exists(meta_path), f"'{meta_path}' dose not exit, run prepare.py first"
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
gptconf.vocab_size = meta['vocab_size']
stoi = meta['stoi']
itos = meta['itos']

# define encoder and decoder
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# initialize model
model = GPT(gptconf)
model.load_state_dict(torch.load(path2state_dict))
model = model.to(device)

# generate from the model
print("start  generate")
print("---------------")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
print("---------------")
print("finish generate")
