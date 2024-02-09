import torch
import torch.nn.functional as F

from . import inference

class SparseCoder(torch.nn.Module):
    """
    Sparse Coding model in PyTorch

    Attributes:
        shape (torch.Size): shape of the input data. 
        num_basis (int): number of basis functions.
        dim_basis (int): dimension of a basis function.
        method (str): method used for sparse coding.
        phi (torch.nn.Parameter): Learnable parameter representing the basis functions.
            Initialized with normalized random values for encoding features.
        device (torch.device): device on which the tensors are placed.
    """
    def __init__(self, num_basis: int, shape: torch.Size, method: str = 'FISTA', device: torch.device = None) -> None:
        """
        Constructor for SparseCoder.

        Parameters:
            num_basis (int): number of basis functions components (phi). 
            shape (torch.Size): shape of the input data. 
            method (str, optional): algorithm used to infer the sparse coeffiecents (alpha). The available methods: {'FISTA', 'ISTA', 'log'} (default: 'FISTA'). 
            device (torch.device, optional): device to place tensors on (default: None).
        """
        if not isinstance(num_basis, int) or not isinstance(shape, torch.Size): 
            raise TypeError("SparseCoder.__init__() arguments (position 1 and 2) must be integers.")
        
        if method not in inference.Methods.__members__:
            raise TypeError(f"SparseCoder.__init__() invalid value for 'method': allowed values are {tuple(inference.Methods.__members__.keys())}.")
        
        super(SparseCoder, self).__init__()
        self.device = device

        self.method = method
        self.shape = shape
        self.num_basis = num_basis
        self.dim_basis = shape.numel()

        self.__phi = torch.nn.Parameter(
            data=F.normalize(
                    torch.randn(self.num_basis, self.dim_basis, device=device), 
                    p=2, 
                    dim=1),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for SparseCoder.

        Parameters:
            x (torch.Tensor): input data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the sparse coefficients (alpha) and the reconstructed image.
        """
        if isinstance(x, torch.Tensor): 
            x.to(self.device)
            x = torch.flatten(x, start_dim=1)
        else:
            raise TypeError("SparseCoder(): argument 'x' (position 1) must be a Tensor.")

        if x.dim() != 2 or x.shape[1] != self.dim_basis: 
            raise ValueError(f"SparseCoder(): input tensor must have shape (batch_size, {self.dim_basis}): got shape {tuple(x.shape)}.")
        
        if self.method == 'FISTA':
            alpha = inference.FISTA(x, phi=self.__phi.detach(), device=self.device)  
        elif self.method == 'ISTA':
            alpha = inference.ISTA(x, phi=self.__phi.detach(), device=self.device)  
        elif self.method == 'LOG':
            alpha = inference.LOG_REGU(x, phi=self.__phi.detach(), device=self.device)

        recon = torch.mm(alpha, self.__phi).view(-1, self.shape[0], self.shape[1])
        return alpha, recon

    @property
    def phi(self) -> torch.Tensor:
        """
        Get the basis functions elements (phi) of the Sparse Coding model.

        Returns:
            torch.Tensor: the set of basis functions.
        """
        return self.__phi.detach().view(-1, self.shape[0], self.shape[1])

