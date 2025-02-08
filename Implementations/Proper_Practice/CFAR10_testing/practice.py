import torch

# Define each tensor
A = torch.tensor([21, 30, 10]) / 255
B = torch.tensor([21, 30, 27]) / 255
C = torch.tensor([21, 10, 30]) / 255
D = torch.tensor([21, 10, 27]) / 255
E = torch.tensor([21, 27, 30]) / 255
F = torch.tensor([21, 27, 10]) / 255
G = torch.tensor([30, 21, 10]) / 255
H = torch.tensor([27, 30, 10]) / 255




print(H+(D-H)*(7/22))



print(H+(E-H)*(8/18))


print(F+(E-F)*(23/33))




print(F+(E-F*(7/33)))



print(B+(C-B)*(13/32))



print(A+(B-A)*(4/18))



print(A+(F-A)*(14/33))

