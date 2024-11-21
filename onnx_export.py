'''https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html'''

# pip install onnx onnxscript
# pip install google protobuf # ModuleNotFoundError: No module named 'google.protobuf.json_format'

import torch
from gpt import GPT
from config import GPTConfig

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
path2ckpt = "./out/ckpt_iter_50000.pt"
path2onnx = "./onnx/gpt_generate.onnx"

# get the configurations
checkpoint = torch.load(path2ckpt, map_location=device)
checkpoint_model_args = checkpoint['model_args']

model_args = dict()
for key in ['n_layer', 'n_head', 'embed_dim', 'time_step', 'bias', 'vocab_size']:
    model_args[key] = checkpoint_model_args[key]
gptconf = GPTConfig(**model_args)
gptconf.batch_size = 1

# create the model
model = GPT(gptconf)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.to(device)

# dummy input
B, T = gptconf.batch_size, gptconf.time_step
print(f"gptconf.batch_size = {gptconf.batch_size}")
dummy_input = torch.randint(low=0, high=gptconf.vocab_size, size=(B, T), device=device)
print(f"dummy_input: {dummy_input.shape}")
max_new_tokens = torch.tensor(500)
args = (dummy_input, max_new_tokens)

# print(model.generate(dummy_input, max_new_tokens=500)[0].tolist())

# convert a torchmodule to a torchscript
# model = torch.jit.trace(func=model.generate, example_inputs=args)
# model = torch.jit.script(model) # with control flow
model.eval()


# Export the model
torch.onnx.export(model=model.generate,               # model being run
                  args=args,                 # model input (or a tuple for multiple inputs)
                  f=path2onnx,               # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input', 'tokens'], # the model's input names
                  output_names = ['output'],         # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
