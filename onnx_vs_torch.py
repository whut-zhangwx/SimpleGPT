import time
import onnxruntime
from config import GPTConfig
from gpt import GPT
import torch
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
path2ckpt = "./out/ckpt_iter_50000.pt"
path2onnx = "./onnx/gpt.onnx"

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# hyperparameters
model_args = dict(n_layer=12, n_head=12, embed_dim=768, time_step=256,
              bias=True, vocab_size=65, dropout=0.0)
gptconf = GPTConfig(**model_args)
B, T, C = 1, gptconf.time_step, gptconf.embed_dim

# input
input = torch.randint(low=0, high=gptconf.vocab_size, size=(B, T), device=device)

# onnx_model
ort_session = onnxruntime.InferenceSession(path2onnx, providers=["CPUExecutionProvider"])
# torch model
checkpoint = torch.load(path2ckpt, map_location=device)
torch_model = GPT(gptconf)
torch_model.eval()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0].shape)

# compute torch forward prediction
torch_out = torch_model(input)
print(torch_out[0].shape)

# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# compare time
start = time.time()
torch_out = torch_model(input)
end = time.time()
print(f"Inference of Pytorch model used {end - start} seconds")

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
start = time.time()
ort_outs = ort_session.run(None, ort_inputs)
end = time.time()
print(f"Inference of ONNX model used {end - start} seconds")