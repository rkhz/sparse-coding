import torch
import torch.nn.functional as F

class Postiveshrink(torch.nn.Module):
    def __init__(self,  lambd: float = 0.5):
        super(Postiveshrink, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return F.relu(x - self.lambd)
    