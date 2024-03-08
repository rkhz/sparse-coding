import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum

from . import nn

class Methods(Enum):
    """
    Enumeration of available methods to infer the sparse coeffiecents (alpha) of a Sparse Coding model.

    Values:
        FISTA (str): Fast Iterative Shrinkage-Thresholding Algorithm.
        ISTA (str): Iterative Shrinkage-Thresholding Algorithm.
        LOG (str): Log-penalty regularization
    """
    FISTA = 'FISTA'
    ISTA = 'ISTA'
    LOG = 'LOG'

def get_Lipschitz_cst(a: torch.Tensor) -> float:
    """
    Find the Lipschitz constant of a matrix by approximating the largest eigenvalue using the LOBPCG method.

    Parameters:
        a (torch.Tensor): Input matrix.

    Returns:
        float: Lipschitz constant.
    """
    if a.shape[0] < a.shape[1]:
        e_val, _ = torch.lobpcg(torch.mm(a, a.T), k=1, largest=True)
    else:
        e_val, _ = torch.lobpcg(torch.mm(a.T, a), k=1, largest=True)
    return e_val.item()

################################
# Laplacian prior (L1 penalty) #
################################
def FISTA(x: torch.Tensor, phi: torch.Tensor, lambda_sparse: float = 0.9, iter: int = 30, only_pos: bool=True,  device: torch.device = None) -> torch.Tensor:
    """
    Infer the sparse coefficients (alpha) of a Sparse Coding model 
    by applying Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

    Parameters:
        x (torch.Tensor): input data.
        phi (torch.Tensor): set of basis functions.
        lambda_sparse (float, optional): sparsity regularization parameter (default: 0.9).
        iter (int, optional): number of iterations (default: 30).
        only_pos (bool, optional): returns only positive coefficents.
        device (torch.device, optional): device to place tensors on (default: None).

    Returns:
        torch.Tensor: sparse coefficients.
    """
    l = get_Lipschitz_cst(phi)

    if not only_pos:
        shrink = torch.nn.Softshrink(lambda_sparse/l)
    else:
        shrink = nn.Postiveshrink(lambda_sparse/l)

    alpha = torch.zeros((x.shape[0], phi.shape[0]), device=device)
    t_curr = t_next = 1
    for _ in range(iter):
        t_curr = t_next
        t_next = (1 + np.sqrt(1 + (4 * t_curr**2))) / 2   
        
        alpha_prev = alpha.clone()

        alpha -= (1/l) * grad_alpha_squared_error(x, alpha, phi)
        alpha = shrink(alpha) #F.softshrink(alpha, lambda_sparse/l) 
        alpha += ((t_curr - 1) / t_next) * (alpha - alpha_prev)
    return alpha


def ISTA(x: torch.Tensor, phi: torch.Tensor, lambda_sparse: float = 0.9, iter: int = 30, device: torch.device = None) -> torch.Tensor:
    """
    Infer the sparse coefficients (alpha) of a Sparse Coding model 
    by applying Iterative Shrinkage-Thresholding Algorithm (ISTA).

    Parameters:
        x (torch.Tensor): input data.
        phi (torch.Tensor): set of basis functions.
        lambda_sparse (float, optional): sparsity regularization parameter (default: 0.9).
        iter (int, optional): number of iterations (default: 30).
        device (torch.device, optional): device to place tensors on (default: None).

    Returns:
        torch.Tensor: sparse coefficients.
    """
    l = get_Lipschitz_cst(phi)
    alpha = torch.zeros((x.shape[0], phi.shape[0]), device=device)
    for _ in range(iter):
        alpha -= (1/l) * grad_alpha_squared_error(x, alpha, phi)
        alpha = F.softshrink(alpha, lambda_sparse/l) 
    return alpha

################################
#  Cauchy prior (Log-penalty)  #
################################
def LOG_REGU(x: torch.Tensor, phi: torch.Tensor, lambda_sparse: float = 1.0, iter: int = 30, device: torch.device = None) -> torch.Tensor:
    """
    Infer the sparse coefficients (alpha) of a Sparse Coding model 
    by applying gradient descent with log-penalty regularization.

    Parameters:
        x (torch.Tensor): input data.
        phi (torch.Tensor): set of basis functions.
        lambda_sparse (float, optional): sparsity regularization parameter (default: 1.0).
        iter (int, optional): number of iterations (default: 30).
        device (torch.device, optional): device to place tensors on (default: None).

    Returns:
        torch.Tensor: sparse coefficients.
    """
    l = get_Lipschitz_cst(phi)
    alpha = torch.zeros((x.shape[0], phi.shape[0]), device=device)
    for _ in range(iter):
        grad = grad_alpha_squared_error(x, alpha, phi) + (lambda_sparse * grad_alpha_log_penalty(alpha))
        alpha -= (1/l) * grad
    return alpha


################################
#      Gradient of alpha       #
################################
def grad_alpha_squared_error(target: torch.Tensor, alpha: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of the squared reconstruction error with respect to sparse coefficients (alpha) 
    for a Sparse Coding model.

    Parameters:
        x (torch.Tensor): input data.
        alpha (torch.Tensor): sparse coefficients.
        phi (torch.Tensor): set of basis functions.

    Returns:
        torch.Tensor: gradient of the squared reconstruction error with respect to sparse coefficients.
    """
    recon = torch.mm(alpha, phi)
    residual = recon - target
    return  torch.mm(residual, phi.T)  

def grad_alpha_log_penalty(alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of the log-penalty regularization term with respect to sparse coefficients (alpha)
    for a Sparse Coding model.

    Parameters:
        alpha (torch.Tensor): sparse coefficients.

    Returns:
        torch.Tensor: gradient of the log-penalty term with respect to sparse coefficients.
    """
    return torch.div(2*alpha, 1 + alpha**2) 