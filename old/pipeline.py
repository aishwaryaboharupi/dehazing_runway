import torch

print("Is PyTorch working?", torch.__version__)
print("Can it see your RTX 4050?", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")