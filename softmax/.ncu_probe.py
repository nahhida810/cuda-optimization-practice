import torch
x = torch.randn(4096, 1024, device='cuda')
for _ in range(20):
    y = torch.softmax(x, dim=-1)
torch.cuda.synchronize()
print('done')
