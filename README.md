## OpenIFEM-SABLE
- A specialized OpenIFEM module designed for the simulation of solid-solid interactions, high-velocity impacts
- A simulation is conducted by running OpenIFEM and SABLE simultaneously 
- OpenIFEM: An implementation of the modified Immersed Finite Element Method (mIFEM) based on [deal.II](https://www.dealii.org/)
- SABLE: Multi-material Eulerian hydrocode by Sandia National Laboratories
- OpenIFEM provides:
     - Lagrangian solid solvers with linear elastic or hyper elastic material descriptions: `linear_elasticity.cpp`, `hyper_elasticity.cpp`  
     - An MPI-based communication wrapper that exchanges relevant quantities between the two software: `sable_wrapper.cpp`, `mpi_sable_wrapper.cpp` 
     - An interaction module based on mIFEM algorithm: `openifem_sable_fsi.cpp`, `mpi_openifem_sable_fsi.cpp`
- SABLE provides:
     - Multi-material domain described using a fixed Eulerian mesh
- For details, please refer [techincal repot](https://www.osti.gov/biblio/1888360) 

## OpenIFEM Dependencies
1. [MPICH](https://www.mpich.org/)/[Open MPI](https://www.open-mpi.org/)
2. [PETSc](https://www.mcs.anl.gov/petsc/) with `MUMPS` and `Hypre`
3. [p4est](http://www.p4est.org/)
4. [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)
5. [deal.II](https://www.dealii.org/) greater than v9.1 with 1-4

## Install

## References
1. @article{cheng2019openifem,
     title={Openifem: a high performance modular open-source software of the immersed finite element method for fluid-structure interactions},
     author={Cheng, Jie and Yu, Feimi and Zhang, Lucy T},
     journal={Computer Modeling in Engineering \& Sciences},
     volume={119},
     number={1},
     pages={91--124},
     year={2019}
   }
2. @article{zhang2004immersed,
     title={Immersed finite element method},
     author={Zhang, Lucy and Gerstenberger, Axel and Wang, Xiaodong and Liu, Wing Kam},
     journal={Computer Methods in Applied Mechanics and Engineering},
     volume={193},
     number={21-22},
     pages={2051--2067},
     year={2004},
     publisher={Elsevier}
   }

3. @article{zhang2007immersed,
     title={Immersed finite element method for fluid-structure interactions},
     author={Zhang, LT and Gay, M},
     journal={Journal of Fluids and Structures},
     volume={23},
     number={6},
     pages={839--857},
     year={2007},
     publisher={Elsevier}
   }
   
4. @article{wang2012semi,
     title={Semi-implicit formulation of the immersed finite element method},
     author={Wang, Xingshi and Wang, Chu and Zhang, Lucy T},
     journal={Computational Mechanics},
     volume={49},
     number={4},
     pages={421--430},
     year={2012},
     publisher={Springer}
   }

5. @article{wang2013modified,
     title={Modified immersed finite element method for fully-coupled fluid--structure interactions},
     author={Wang, Xingshi and Zhang, Lucy T},
     journal={Computer methods in applied mechanics and engineering},
     volume={267},
     pages={150--169},
     year={2013},
     publisher={Elsevier}
   }
