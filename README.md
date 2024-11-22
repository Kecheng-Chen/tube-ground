# Coupling between 1D tube and 3D ground

### Prerequisite packages
> The following prerequisite packages can be found in the docker image:

* [Boost](http://www.boost.org/)
* [Dealii](https://dealii.org)

## Compile
> See https://thm-doc.cb-geo.com/ for more detailed instructions. 

0. Run `mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++  -DDEAL_II_DIR=/path/to/dealii/ ..`

1. Run `make clean && make -jN` (where N is the number of cores).


## Compile and Run on TACC

```

#login TACC
ssh taccuserid@ls5.tacc.utexas.edu

module load impi-largemem
export CC=icc
export CXX=icpc

# Compile PETSc 3.13
wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.13.3.tar.gz 
tar -xvzf petsc-3.13.3.tar.gz
cd petsc-3.13.3/
export PETSC_ARCH=clx
export PETSC_DIR=$HOME/petsc-3.13.3
./config/configure.py --with-shared=1 --with-x=0 --with-mpi=1 --with-debugging=0 --with-blas-lapack-dir=$TACC_MKL_LIB -COPTFLAGS=O2 -CXXOPTFLAGS=O2 -FOPTFLAGS=O2
make PETSC_DIR=$HOME/petsc-3.13.3 PETSC_ARCH=clx all -j4
make PETSC_DIR=$HOME/petsc-3.13.3 PETSC_ARCH=clx check -j4

# METIS
cd $HOME
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz && \
    tar -xf metis-5.1.0.tar.gz && \
    cd metis-5.1.0/ && \
    make config shared=1 cc=icc cxx=icpc prefix=$HOME/metis && \
    make install -j4 && cd ..

# Load module boost
module load boost

# Clone and compile dealii
cd $HOME
git clone https://github.com/dealii/dealii --depth=1 dealii-src
cd dealii-src/ && mkdir build &&  cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/dealii -DPETSC_DIR=$PETSC_DIR -DPETSC_ARCH=clx -DDEAL_II_WITH_PETSC=On  -DDEAL_II_WITH_METIS=On -DMETIS_DIR=$HOME/metis/ -DDEAL_II_WITH_MPI=On ..
make install -j4


# Clone THM
cds
git clone https://github.com/cb-geo/thm
cd thm
git checkout thm_seg_parallel
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$HOME/dealii/ ..
make -j
```

To run on TACC on multiple nodes create a submission script
```
touch submit
```
and update the submission script as:

```
#!/bin/bash
#SBATCH -J thm-N1n2      # job name
#SBATCH -o thm-N1n2.o%j  # output and error file name (%j expands to jobID)
#SBATCH -A Material-Point-Metho # Project
#SBATCH -N 1             # number of nodes requested
#SBATCH -n 2             # total number of mpi tasks requested
#SBATCH -p normal     # queue (partition) -- normal, development, etc.
#SBATCH -t 00:15:00       # run time (hh:mm:ss) - 10 hours
# Slurm email notifications are now working on Lonestar 5
#SBATCH --mail-type=fail   # email me when the job fails
# run the executable named a.out
module load boost
export CC=icc
export CXX=icpc
ibrun ./thm
```

Then submit the job using: `sbatch submit`

sbatch submit

```

# This command offers similar functions as showq, but has more options. To monitor the 
# statues of the jobs you have submitted, use 

squeue -u CRSid 

# or use 

showq -u 

# to monitor status of the jobs.

# If want to cancel the squeue

scancel <jobid>

# if want to see the utilization of computer
wwall -j <jobid>
# to see the path of scratch
echo $SCRATCH

# to see the data folder
cd $SCRATCH
