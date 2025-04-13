import torch
import torchtext

print(torch.__version__)                # 2.3.0+cu121
print(torchtext.__version__)           # 0.18.0
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # RTX 2070