# PyTorch - Sparse Coding 
 PyTorch implementation, with CUDA support, of the sparse coding algorithm based on the paper by Olshausen and Field (1997). Inference is performed using either the FISTA (Fast Iterative Shrinkage-Thresholding Algorithm), ISTA (Iterative Shrinkage-Thresholding Algorithm) algorithm, or gradient descent with a log-penalty regularization (differentiable).

![Example GIF](example.gif)

## Installation
- Clone the repository: `git clone https://github.com/rkhz/sparse-coding.git`
- Install dependencies: `pip install -r requirements.txt`


## References
1. **Sparse Coding with an Overcomplete Basis Set: A Strategy Employed by V1?**
   - Bruno A. Olshausen and David J. Field
   - Vision Research, Vol. 37, No. 23, pp. 3311–3325, 1997.
   - DOI: [10.1016/s0042-6989(97)00169-7](https://doi.org/10.1016/s0042-6989(97)00169-7)
     
2. **An iterative thresholding algorithm for linear inverse problems with a sparsity constraint**
   - Ingrid Daubechies, Michel Defrise and Christine De Mol
   - Communications on Pure and Applied Mathematics, Vol. 57, No. 11, pp. 1413-1457, 2004.
   - DOI: [10.1002/cpa.20042](https://doi.org/10.1002/cpa.20042)
     
3. **A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems**
   - Amir Beck and Marc Teboulle
   - SIAM Journal on Imaging Sciences, Vol. 2, No. 1, pp. 183–202, 2009.
   - DOI: [10.1137/080716542](https://doi.org/10.1137/080716542)
