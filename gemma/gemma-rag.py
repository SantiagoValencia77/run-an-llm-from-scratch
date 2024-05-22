import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} ON!")
else:
    print("GPU Off.")