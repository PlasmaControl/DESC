# DESC Apptainer Containers

This document provides instructions for using DESC with Apptainer/Singularity containers, particularly useful for HPC clusters and reproducible research environments.

## Overview

DESC provides pre-built Apptainer containers that include:
- Complete DESC installation with all dependencies
- CPU and GPU variants
- Multiple Python versions (3.10, 3.11, 3.12, 3.13)
- Optimized for HPC cluster environments
- Ready-to-use examples and test cases

## Available Container Images

### Production Images (from releases)
```bash
# Latest release - CPU version
ghcr.io/tokamaster/desc:latest

# Latest release - GPU version  
ghcr.io/tokamaster/desc:gpu

# Specific version - CPU
ghcr.io/tokamaster/desc:v0.12.2

# Specific version - GPU
ghcr.io/tokamaster/desc:v0.12.2-gpu
```

### Development Images
```bash
# Development branch - CPU
ghcr.io/tokamaster/desc:dev-cpu

# Development branch - GPU
ghcr.io/tokamaster/desc:dev-gpu
```

## Quick Start

### 1. Pull a Container
```bash
# CPU version (most common)
apptainer pull desc-cpu.sif oci://ghcr.io/tokamaster/desc:latest

# GPU version (requires NVIDIA GPU)
apptainer pull desc-gpu.sif oci://ghcr.io/tokamaster/desc:gpu
```

### 2. Run DESC
```bash
# Interactive Python session
apptainer run desc-cpu.sif

# Run a DESC example
apptainer exec desc-cpu.sif python -m desc desc/examples/SOLOVEV

# Shell access
apptainer shell desc-cpu.sif
```

## Detailed Usage

### Basic Commands

#### Interactive Python Session
```bash
apptainer run desc-cpu.sif
# This starts Python with DESC pre-loaded and ready to use
```

#### Execute Python Scripts
```bash
# Run your own script
apptainer exec desc-cpu.sif python your_script.py

# Run with specific arguments
apptainer exec desc-cpu.sif python -m desc --verbose your_input.txt
```

#### Shell Access
```bash
# Get a shell inside the container
apptainer shell desc-cpu.sif

# Inside the container, DESC is ready to use:
$ python -c "import desc; print(desc.__version__)"
$ python -m desc desc/examples/SOLOVEV
```

### Working with Files

#### Bind Local Directories
```bash
# Bind your current directory to access local files
apptainer exec --bind $(pwd):/work desc-cpu.sif python /work/my_script.py

# Bind multiple directories
apptainer exec --bind /home/user/data:/data,/scratch:/scratch desc-cpu.sif python script.py
```

#### Home Directory Access
By default, Apptainer binds your home directory, so files in `$HOME` are accessible:
```bash
# Files in your home directory are automatically available
apptainer exec desc-cpu.sif python ~/my_desc_script.py
```

### GPU Usage

For GPU-enabled containers, use the `--nv` flag to enable NVIDIA GPU access:

```bash
# Pull GPU container
apptainer pull desc-gpu.sif oci://ghcr.io/tokamaster/desc:gpu

# Run with GPU access
apptainer exec --nv desc-gpu.sif python your_gpu_script.py

# Verify GPU access
apptainer exec --nv desc-gpu.sif python -c "import jax; print(jax.devices())"
```

## HPC Cluster Usage

### SLURM Example

Create a SLURM batch script (`desc_job.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=desc-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=16G

# For GPU jobs, add:
#SBATCH --gres=gpu:1

# Load Apptainer module (if needed)
module load apptainer

# Set number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run DESC
apptainer exec desc-cpu.sif python -m desc input_file.txt

# For GPU version:
# apptainer exec --nv desc-gpu.sif python -m desc input_file.txt
```

Submit the job:
```bash
sbatch desc_job.slurm
```

### PBS Example

Create a PBS script (`desc_job.pbs`):

```bash
#!/bin/bash
#PBS -N desc-job
#PBS -l nodes=1:ppn=8
#PBS -l walltime=02:00:00
#PBS -l mem=16gb

# For GPU:
#PBS -l nodes=1:ppn=8:gpus=1

cd $PBS_O_WORKDIR

# Load Apptainer (if needed)
module load apptainer

# Run DESC
apptainer exec desc-cpu.sif python -m desc input_file.txt
```

Submit the job:
```bash
qsub desc_job.pbs
```

### Interactive Jobs

Request an interactive session with Apptainer:

```bash
# SLURM interactive session
salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --time=1:00:00

# Once allocated, run the container
apptainer shell desc-cpu.sif
```

## Advanced Usage

### Custom Environment Variables
```bash
# Set environment variables for the container
apptainer exec --env JAX_ENABLE_X64=True,OMP_NUM_THREADS=4 desc-cpu.sif python script.py
```

### Memory and Performance Tuning
```bash
# Set memory limits and thread counts
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
apptainer exec desc-cpu.sif python script.py
```

### Building Custom Containers

If you need to modify the container, you can build from the provided definition:

```bash
# Clone DESC repository
git clone https://github.com/tokamaster/DESC.git
cd DESC

# Build CPU version
apptainer build --fakeroot desc-custom.sif apptainer.def

# Build GPU version
apptainer build --build-arg GPU=true --fakeroot desc-gpu-custom.sif apptainer.def
```

## Examples

### Example 1: Run SOLOVEV equilibrium
```bash
apptainer exec desc-cpu.sif python -m desc -vv desc/examples/SOLOVEV
```

### Example 2: Python script with DESC
Create `test_desc.py`:
```python
import desc
from desc.equilibrium import Equilibrium
from desc.profiles import PowerSeriesProfile

# Create pressure and rotational transform profiles
pres = PowerSeriesProfile([1e3, 0, -2e3])
iota = PowerSeriesProfile([1.0, 0.1, 0])

# Create equilibrium
eq = Equilibrium(M=3, N=2, pressure=pres, iota=iota)
print(f"Created equilibrium with {eq.M} poloidal modes and {eq.N} toroidal modes")

# Solve equilibrium
eq.solve(verbose=2)
print("Equilibrium solved successfully!")
```

Run it:
```bash
apptainer exec desc-cpu.sif python test_desc.py
```

### Example 3: Batch Processing
Create `batch_process.py`:
```python
import desc
import sys

# Process multiple input files
input_files = sys.argv[1:]
for input_file in input_files:
    print(f"Processing {input_file}...")
    # Your DESC processing code here
```

Run on multiple files:
```bash
apptainer exec --bind $(pwd):/work desc-cpu.sif python /work/batch_process.py file1.txt file2.txt
```

## Troubleshooting

### Common Issues

1. **Permission denied errors**:
   ```bash
   # Make sure your script is executable
   chmod +x your_script.py
   
   # Or run with python explicitly
   apptainer exec desc-cpu.sif python your_script.py
   ```

2. **File not found errors**:
   ```bash
   # Use absolute paths or bind directories
   apptainer exec --bind $(pwd):/work desc-cpu.sif python /work/script.py
   ```

3. **Module not found errors**:
   ```bash
   # Verify container contents
   apptainer exec desc-cpu.sif python -c "import desc; print(desc.__version__)"
   ```

4. **GPU not detected**:
   ```bash
   # Make sure to use --nv flag and GPU container
   apptainer exec --nv desc-gpu.sif python -c "import jax; print(jax.devices())"
   ```

### Getting Help

1. **Check container info**:
   ```bash
   apptainer inspect desc-cpu.sif
   ```

2. **View container help**:
   ```bash
   apptainer run-help desc-cpu.sif
   ```

3. **Test container functionality**:
   ```bash
   apptainer test desc-cpu.sif
   ```

## Container Information

The containers include:
- **Base OS**: Ubuntu 22.04
- **Python**: 3.10, 3.11, 3.12, or 3.13 (depending on variant)
- **DESC**: Latest version with all dependencies
- **JAX**: CPU or GPU backend
- **Scientific libraries**: NumPy, SciPy, Matplotlib, H5Py, etc.
- **Size**: ~2-4 GB (varies by configuration)

For more information, visit the [DESC documentation](https://desc-docs.readthedocs.io/) or [GitHub repository](https://github.com/tokamaster/DESC).
