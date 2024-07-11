import torch
import argparse
import pathlib
import logging

from gpt import GPTLanguageModel

def get_parser():
    parser = argparse.ArgumentParser(description="hyper parameters for GPT")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--time_step', default=256, type=int)
    parser.add_argument('--embed_dim', default=384, type=int)
    parser.add_argument('--max_iters', default=5000, type=int)
    parser.add_argument('--eval_interval', default=500, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--eval_iters', default=200, type=int)
    parser.add_argument('--n_head', default=6, type=int)
    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--path2state_dict', default="./state_dict/",type=pathlib.Path)
    return parser


# data loading
def get_batch(split, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - args.time_step, (args.batch_size,))
    x = torch.stack([data[i:i+args.time_step] for i in ix])
    y = torch.stack([data[i+1:i+args.time_step+1] for i in ix])
    x, y = x.to(args.device), y.to(args.device)
    return x, y

@torch.no_grad()
def estimate_loss(eval_iters, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    
    # get hyperparameters
    parser = get_parser()
    args = parser.parse_args([
        "--batch_size", "64",
        "--time_step", "256",
        "--embed_dim", "384",
        # "--max_iters", "5000",
        "--max_iters", "2",
        "--eval_interval", "500",
        "--learning_rate", "3e-4",
        "--eval_iters", "200",
        "--n_head", "6",
        "--n_layer", "6",
        "--dropout", "0.2",
        "--pretrain", "True",
        "--path2state_dict", "./state_dict/gpt.pth",
    ])
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"device: {args.device}")

    # print(args.device)
    # print(args.path2state_dict)
    # print(args.dropout, type(args.dropout))
    # print(args.learning_rate, type(args.learning_rate))

    torch.manual_seed(1337)

    logging.info(f"loading dataset")
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    args.vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    
    logging.info("initializing model")
    # initialize model
    model = GPTLanguageModel(args)
    if args.pretrain :
        logging.info(f"loading state_dict")
        model.load_state_dict(torch.load(args.path2state_dict))
    model = model.to(args.device)
    # print the number of parameters in the model
    logging.info(f"{sum([p.numel() for p in model.parameters()])/1e6} M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    logging.info(f"start iteration")
    for iter in range(args.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss(args.eval_interval, args.device)
            logging.info(f"iter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', args.device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    logging.info(f"finish iteration")
    logging.info(f"start saving model state_dict")
    torch.save(model.state_dict(), args.path2state_dict)
    logging.info(f"save model state_dict to '{args.path2state_dict}'")
    logging.info("END")

    # # generate from the model
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    # #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
