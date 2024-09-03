import os
import logging
import pickle

import torch
import numpy


from gpt_old import GPT
# from model import GPT
from config import GPTConfig

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # load a huge data file (on disk) into memory batch by batch using memory mapping
    # https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
    # memory map ./tinyshakespeare/train.bin or ./tinyshakespeare/val.bin
    data = numpy.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=numpy.uint16, mode='r')
    idx = torch.randint(len(data) - gptconf.time_step, (gptconf.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+gptconf.time_step]).astype(numpy.int64)) for i in idx])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+gptconf.time_step]).astype(numpy.int64)) for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# set logging config
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

# set train hyperparameters
max_iters = 2
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
use_pretrain = True
path2state_dict = "./state_dict/gpt.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"device: {device}")

# set dataset
data_dir = './tinyshakespeare'

# get model config
gptconf = GPTConfig()

# set random seed
torch.manual_seed(1337)

logging.info(f"loading dataset")
# run ./tinyshakespeare/prepare.py to obtain meta.pkl, train.bin and val.bin
meta_path = os.path.join(data_dir, 'meta.pkl')
assert os.path.exists(meta_path), f"'{meta_path}' dose not exit, run prepare.py first"
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
gptconf.vocab_size = meta['vocab_size']
stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

logging.info("initializing model")
# initialize model
model = GPT(gptconf)
if use_pretrain:
    logging.info(f"loading state_dict")
    model.load_state_dict(torch.load(path2state_dict))
model = model.to(device)
# print the number of parameters in the model
logging.info(f"{sum([p.numel() for p in model.parameters()])/1e6} M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

logging.info(f"start iteration")
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(eval_interval)
        logging.info(f"iter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

logging.info(f"finished iteration")
logging.info(f"start saving model state_dict")
torch.save(model.state_dict(), path2state_dict)
logging.info(f"saved model state_dict to '{path2state_dict}'")

# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
