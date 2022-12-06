This folder contains a Jupyter notebook to reproduce the plots from the paper "The DESC Stellarator Code Suite Part I: Quick and accurate equilibria computations."

Many of the results in the paper rely on numerous DESC and VMEC output files which are quite large and so are not included here. Please see the [Zenodo](https://doi.org/10.5281/zenodo.6539680) for this paper to recreate the full results.

The input files used to obtain these results are included in the Zenodo.
The DESC results can be reproduced by first checking out v0.5.0 of desc, then running `desc <input>` for each input file
To remake the plots, move the notebook inside of this directory into the same directory as the `DESC_outputs` and `VMEC_outputs` folders in the Zenodo folder

The VMEC results can be reproduced by running `xvmec2000 <input>` (assuming you have VMEC version 9.0 installed such as on the PPPL clusters)
Most folders with input files include the SLURM job script `job.slurm` used to run code, and can be ran with (on PPPL clusters) `sbatch job.slurm`

---------------------------------------------------------
Notes on installing DESC v0.5.0: It is recommended to make a new environment as package requirements are different for v0.5.0 versus the current DESC version. The following installation steps were found to work on a Linux machine:

```
conda install pip
conda install python=3.8
```
now split the requirements.txt into pip_only_req.txt containing the nvgpu, jax, jaxlib lines (deleting those lines from requirements.txt)
```
conda install --file requirements.txt
pip install -r pip_only_req.txt
pip install flatbuffers==1.12
conda install jupyterlab
```
---------------------------------------------------------
