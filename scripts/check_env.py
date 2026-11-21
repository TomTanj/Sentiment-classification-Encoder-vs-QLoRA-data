import torch
print("torch        :", torch.__version__)
if not torch.cuda.is_available():
    print("CUDA ready   : False  <-- problem!")
else:
    p = torch.cuda.get_device_properties(0)
    print("CUDA ready   : True")
    print("GPU          :", p.name)
    print("VRAM         :", round(p.total_memory/1024**3, 1), "GB")
    print("Compute cap. :", f"{p.major}.{p.minor}")
    x = torch.randn(1024, 1024, device="cuda")
    print("GPU compute  :", (x @ x).sum().item() != 0)

import transformers, peft, trl, datasets
import bitsandbytes as bnb
print("transformers :", transformers.__version__)
print("peft / trl   :", peft.__version__, "/", trl.__version__)
print("datasets     :", datasets.__version__)
print("bitsandbytes :", bnb.__version__)
