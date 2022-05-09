This folder contains input files and a Jupyter notebook to reproduce the plots from the paper "The DESC Stellarator Code Suite Part I: Quick and accurate equilibria computations." 

Many of the results in the paper rely on numerous DESC and VMEC output files which are quite large and so are not included here. Please see the Zenodo for this paper *INCLUDE ZENODO HERE* to recreate the full results.

The input files used to obtain these results are included. 
The DESC results can be reproduced by running ``desc <input>`` for each input file 
The VMEC results can be reproduced by running ``xvmec2000 <input>`` (assuming you have VMEC installed such as on the PPPL clusters)
Most folders with input files include the SLURM job script ``job.slurm`` used to run code, and can be ran with (on PPPL clusters) ``sbatch job.slurm`` 