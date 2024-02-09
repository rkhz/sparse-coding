import torch
import torch.nn.functional as F

class SGD_L2Norm(torch.optim.SGD):
    """
    SGD_L2Norm extends the functionality of torch.optim.SGD by only adding a 
    L2 normalization to the parameters after each optimization step.

    Attributes:
        Inherits attributes from torch.optim.SGD.

    Example:
        >>> optimizer = SGD_L2Norm(model.parameters(), lr=1e-2, momentum=0.9)
        >>> optimizer.step()
    """

    def __init__(self, params, lr=1e-2, momentum=0):
        """
        Constructor for SGD_L2Norm.

        Parameters:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): learning rate (default: 1e-2).
            momentum (float, optional): Momentum factor (default: 0).
        """
        super(SGD_L2Norm, self).__init__(params, lr=lr, momentum=momentum)

    def step(self, closure=None):
        """
        Performs a single optimization step: in addition to the default optimization step from 
        the parent class torch.optim.SGD, this method applies L2 normalization to the parameters.

        Parameters:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        # optimization step
        super(SGD_L2Norm, self).step(closure)

        # normalization step
        for group in self.param_groups:
            for params in group['params']:
                params.data = F.normalize(params.data, p=2, dim=1)