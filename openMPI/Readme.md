# Elastic Net for Structural Equation Models (SEM)

Anhui Huang | Ph.D. Electrical and Computer Engineering 

<https://scholar.google.com/citations?user=WhDMZEIAAAAJ&hl=en>

## How to use openMPI for parallel computation 

### dependencies
The compilation of the C/C++ package depends on the following packages:
- `intel-mkl`
- `gcc`
- `openmpi/gcc`

### compile
`mpicc -o enSEMmpiV1_3 enSEMmpiV1_3.c -L/share/apps/intel/Compiler/11.1/069/mkl/lib/em64t -lmkl_intel_lp64 -lmkl_sequential -lmkl_core  -I/share/apps/intel/Compiler/11.1/069/mkl/include`

### run
`mpirun -np 8 enSEMmpiV1_3 -n_samples 200 -n_nodes 30 -verbose 3 -response Yn200Ng30.dat -covariate Xn200Ng30.dat -true Bn200Ng30.dat -missing Mn200Ng30.dat`

- `-np`: number of processors to run the job
- `-n_samples`: sample size `N`
- `-n_nodes`: network node size `M`
- `-verbose`: amount of output 
- `-response`: the input matrix `Y` (M by N size)
- `-covaraite`: the input matrix `X` (M by N size)
- `-true`: the input matrix `B` (see package description if this is real data without known network structure)
- `-missing`: the input matrix `M`(see package description if there is no missing values)

## Yeast Gene Regulatory Network
To run the yeast GRN, copy the two data files from `data/` folder to the working directory:
- `data/Yeast_Y_file.txt`
- `data/Yeast_X_file.txt`

See files in the `doc/` folder for data preprocessing details. The network can be obtained by running the following: 
`mpirun -np 60 enSEMmpiV1_3 -n_samples 112 -n_nodes 3384 -verbose 3 -response Yeast_Y_file.txt -covariate Yeast_X_file.txt -o EN_Yeast_network_result.txt`
