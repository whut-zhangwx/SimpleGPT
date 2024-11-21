'''https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html'''

# pip install onnx onnxruntime

# import onnx
# onnx_model = onnx.load("./onnx/gpt.onnx")
# onnx.checker.check_model(onnx_model)

import onnxruntime
from config import GPTConfig
import torch

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_session = onnxruntime.InferenceSession("./onnx/gpt.onnx", providers=["CPUExecutionProvider"])

model_args = dict(n_layer=12, n_head=12, embed_dim=768, time_step=256,
              bias=True, vocab_size=65, dropout=0.0)

gptconf = GPTConfig(**model_args)
B, T, C = gptconf.batch_size, gptconf.time_step, gptconf.embed_dim

# input
input = torch.randint(low=0, high=gptconf.vocab_size, size=(B, T), device=torch.device('cpu'))

# ort_inputs = ort_session.get_inputs()
# # for item in ort_inputs:
# #     print(item)
# print(ort_inputs[0].name)


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
ort_outs = ort_session.run(None, ort_inputs)

# print(len(ort_outs))
print(ort_outs[0].shape)


# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
# print(ort_outs[0].shape)
# print("Exported model has been tested with ONNXRuntime, and the result looks good!")