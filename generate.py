import torch
import argparse
import pathlib

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
    parser.add_argument('--path2state_dict', default="./state_dict/",type=pathlib.Path)
    return parser

if __name__ == "__main__":
    
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
        "--path2state_dict", "./state_dict/gpt.pth",
    ])
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(1337)

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

    # initialize model
    model = GPTLanguageModel(args)
    model.load_state_dict(torch.load(args.path2state_dict))
    model = model.to(args.device)
    # print the number of parameters in the model

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

    print("FINISH")
