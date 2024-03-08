# Sparse Coding - PyTorch
 PyTorch implementation of sparse coding&mdash;based on Olshausen and Field (1997). This repository offers CUDA-accelerated support.
 
 Inference is performed using either FISTA (Fast Iterative Shrinkage-Thresholding Algorithm), ISTA (Iterative Shrinkage-Thresholding Algorithm), or gradient descent with a log-penalty regularization (differentiable). 

![Example GIF](example.gif)

## Installation
- Clone the repository: `git clone https://github.com/rkhz/sparse-coding.git`
- Install dependencies: `pip install -r requirements.txt`


## References

```
@article{olshausen1997sparsecoding,
   title = {Sparse coding with an overcomplete basis set: A strategy employed by V1?},
   author = {Bruno A. Olshausen and David J. Field},
   journal = {Vision Research},
   volume = {37},
   number = {23},
   pages = {3311-3325},
   year = {1997},
   doi = {https://doi.org/10.1016/S0042-6989(97)00169-7}
}
```

```
@article{daubechies2004ista,
   title = {An iterative thresholding algorithm for linear inverse problems with a sparsity constraint},
   author = {Daubechies, I. and Defrise, M. and De Mol, C.},
   journal = {Communications on Pure and Applied Mathematics},
   volume = {57},
   number = {11},
   pages = {1413-1457},
   year = {2004},
   doi = {https://doi.org/10.1002/cpa.20042}
}
```

```
@article{beck2009fista,
   title = {A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems},
   author = {Beck, Amir and Teboulle, Marc},
   journal = {SIAM Journal on Imaging Sciences},
   volume = {2},
   number = {1},
   pages = {183-202},
   year = {2009},
   doi = {10.1137/080716542}
}
```
