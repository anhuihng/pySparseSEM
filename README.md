# Elastic Net for Structural Equation Models (SEM)

Anhui Huang | Ph.D. Electrical and Computer Engineering 

<https://scholar.google.com/citations?user=WhDMZEIAAAAJ&hl=en>


## PyPI installation 
`sparseSEM` is available on PyPI:  https://pypi.org/project/sparseSEM/1.0/. Run command `pip install sparseSEM` to install 
from PyPI.

`test/` folder contains examples using data packed along with this package in `data/` folder. 
To run `test/` examples, clone this repo, and run from `test/` directory. 


## Documentation
The theory and background for network topology inference using sparse Structural Equation Models (SEM) can be found 
in my Ph.D dissertation (Huang A. 2014). The experimental study are also available in the `doc/` folder in the package.  


## Configuration
This package was originally developed to leverage high performance computer clusters to enable parallel computation 
through openMPI.  Users who have access to large scale computational resources can explore the functionality and 
checkout the openMPI module in this package.

Current package utilizes blas/lapack for high speed computation. To build the C/C++ code, the intel OneMKL library is 
specified in the package setup. 
- Install the free OneMKL package (https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-0/intel-oneapi-math-kernel-library-onemkl.html)
- Check if your package is the same as in the setup.py file ('/opt/intel/oneapi/mkl/2023.1.0/include'). Update the file 
accordingly if it was installed in a different path.


## R package
An R package with similiar implementation is also available at CRAN: https://cran.r-project.org/web/packages/sparseSEM/index.html

## OpenMPI
C/C++ implementation of sparseSEM with openMPI for parallel computation is available in openMPI branch (https://github.com/anhuihng/pySparseSEM/tree/openMPI). 

    
## Reference
    - Huang A. (2014) Sparse Model Learning for Inferring Genotype and Phenotype Associations. Ph.D Dissertation,
    University of Miami, Coral Gables, FL, USA.
    - Huang A. (2014) sparseSEM: Sparse-Aware Maximum Likelihood for Structural Equation Models. Rpackage
    (https://cran.r-project.org/web/packages/sparseSEM/index.html)