import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint = {
#     'state_dict': raw_model.state_dict(),
#     'optimizer': optimizer.state_dict(),
#     'model_args': model_args,
#     'iter_num': iter_num,
#     'best_val_loss': best_val_loss,
#     'config': config,
# }

ckpt_path = "./out/ckpt_iter_50000.pt"

assert os.path.exists(ckpt_path), f"{ckpt_path} doesn't exit."

checkpoint = torch.load(ckpt_path, map_location=device)

model_args = checkpoint['model_args']
print(model_args)

state_dict = checkpoint['state_dict']
for layer_name, weight_matrix in state_dict.items():
  print(f"{layer_name}\t{weight_matrix.shape}")

