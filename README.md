# Elastic Net for Structural Equation Models (SEM)

Anhui Huang | Ph.D. Electrical and Computer Engineering 

<https://scholar.google.com/citations?user=WhDMZEIAAAAJ&hl=en>

## Summary
Provides elastic net penalized maximum likelihood for structural equation models (SEM).   The package implements 
`lasso` and `elastic net` (l1/l2) penalized SEM and 
estimates the model parameters with an efficient block coordinate ascent algorithm that maximizes the penalized 
likelihood of the SEM.  Hyperparameters are inferred from cross-validation (CV).  A Stability Selection (STS) function 
is also available to provide accurate causal effect selection. 

The experimental study and vignettes are also available in the `doc/` folder in the package.  

## PyPI installation 
`sparseSEM` is available on PyPI:  https://pypi.org/project/sparseSEM/. Run command `pip install sparseSEM` to install 
from PyPI.

`test/` folder contains examples using data packed along with this package in `data/` folder. 
To run `test/` examples, clone this repo, and run from `test/` directory. 


### Configuration
This package was originally developed to leverage high performance computer clusters to enable parallel computation 
through openMPI.  Users who have access to large scale computational resources can explore the functionality and 
checkout the openMPI module in this package.

Current package utilizes blas/lapack for high speed computation. To build the C/C++ code, the intel OneMKL library is 
specified in the package setup. 
- Install the free OneMKL package (https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-0/intel-oneapi-math-kernel-library-onemkl.html)
- Check if your package is the same as in the setup.py file ('/opt/intel/oneapi/mkl/2023.1.0/include'). Update the file 
accordingly if it was installed in a different path.

### Release Note
- V2.0: add more output information include CV results, hyperparameter, and details of model fit. V2.0 is a major release with stability selection added.
- V1: initial release with corresponding to R package v2.


## Package for other platforms
### R package
An R package for `sparseSEM` is also available at CRAN: https://cran.r-project.org/web/packages/sparseSEM/index.html

### OpenMPI
C/C++ implementation of sparseSEM with openMPI for parallel computation is available in openMPI branch (https://github.com/anhuihng/pySparseSEM/tree/openMPI). 

### Network Inference Example
The following network was a Yeast Gene Regulatory Network (GRN) inferred by `sparseSEM`. 


![Alt text](doc/Fig4_cisTransNetworkYeast_clusters_manualAdjusted_GOterm.jpg?raw=true "Title")
Fig.1 Sparse budding yeast GRN inferred by `sparseSEM`
![Alt text](doc/Fig5_yeast_grn_genome_biology.png?raw=true "Title")
Fig.2 Budding yeast gene interaction pattern with link edges identified by `sparseSEM`

 Network visualized via circos and cytoscope packages. See `openMPI/readme.md` as well as `doc/*` for more details.

## Reference
    - Huang A. (2014) Sparse Model Learning for Inferring Genotype and Phenotype Associations. Ph.D Dissertation,
    University of Miami, Coral Gables, FL, USA.
    - Huang A. (2014) sparseSEM: Sparse-Aware Maximum Likelihood for Structural Equation Models. Rpackage
    (https://cran.r-project.org/web/packages/sparseSEM/index.html)