import torch
print(torch.cuda.is_available())  # Moet True zijn
print(torch.cuda.get_device_name(0))