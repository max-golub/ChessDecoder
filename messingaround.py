import torch
import torch.nn.functional as F
torch.set_printoptions(threshold=1000)
src = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
out = src[[0,2]]
outs = torch.tensor([[0,2], [1,3]])
out = src[outs]
print(out.tolist())

y = torch.tensor([1,4,9])
x = F.one_hot(y)
print(x)

zeros = torch.tensor([0] * 10)
inds = [0,3,5]
zeros[inds] = 1
print(zeros)

z = torch.tensor([3,6,2,7,2,0,1,4,2,3])
z = torch.masked_fill(z, zeros, -1)
print(z)