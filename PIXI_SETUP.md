# DESC Pixi Environment Setup

This document explains how to set up DESC using Pixi package manager for different computing environments, including local machines and various HPC clusters.

## Quick Start

1. **Install Pixi**: Follow instructions at https://pixi.sh/latest/
2. **Clone DESC and setup**:
   ```bash
   git clone https://github.com/PlasmaControl/DESC.git
   cd DESC
   ```

## Environment Overview

The `pixi.toml` file defines several environments optimized for different use cases:

| Environment | Description | Use Case |
|-------------|-------------|----------|
| `default` | Basic CPU environment | Local development, basic usage |
| `dev` | CPU + development tools | Local development with testing/docs |
| `gpu` | GPU-enabled environment | Local GPU development |
| `gpu-dev` | GPU + development tools | Full GPU development setup |
| `cluster` | Basic cluster environment | HPC clusters, CPU-only |
| `cluster-gpu` | Cluster with GPU support | GPU clusters |
| `cluster-dev` | Cluster + development tools | Development on clusters |
| `cluster-gpu-dev` | Full cluster setup | Full development on GPU clusters |

## Installation Instructions

### Local Machine (CPU)

For basic usage on your local machine:
```bash
# Activate the default environment and setup
pixi shell
pixi run setup-cpu

# Or for development:
pixi shell -e dev
pixi run setup-dev
```

### Local Machine (GPU)

If you have a local GPU (NVIDIA):
```bash
# Basic GPU setup
pixi shell -e gpu
pixi run setup-gpu

# GPU development setup
pixi shell -e gpu-dev
pixi run setup-gpu-dev
```

### Generic HPC Clusters

For most Linux computing clusters:
```bash
# CPU-only cluster setup
pixi shell -e cluster
pixi run setup-cluster-cpu

# GPU cluster setup
pixi shell -e cluster-gpu
pixi run setup-cluster-gpu
```

### Specific Cluster Configurations

#### PPPL/Princeton Clusters
```bash
pixi shell -e cluster-gpu
pixi run setup-princeton
```

#### NERSC
```bash
pixi shell -e cluster-gpu
pixi run setup-nersc
```

#### Generic HPC with Unknown Modules
```bash
pixi shell -e cluster
pixi run setup-hpc
```

## Verification

After setup, verify your installation:
```bash
# Check that DESC is properly installed
pixi run check-install

# Run a test example
pixi run test-example
```

## Development Workflows

### Running Tests
```bash
# All tests (slow)
pixi run test

# Fast tests only
pixi run test-fast

# Unit tests only
pixi run test-unit

# With coverage
pixi run test-coverage
```

### Code Quality
```bash
# Format code
pixi run format

# Check formatting
pixi run format-check

# Lint code
pixi run lint

# Setup pre-commit hooks
pixi run pre-commit-install
```

### Documentation
```bash
# Build documentation
pixi run docs-build

# Clean documentation build
pixi run docs-clean

# Serve documentation locally
pixi run docs-serve
# Then visit http://localhost:8000
```

## Cluster-Specific Notes

### Module Systems

The cluster tasks automatically detect and load common modules:
- `anaconda`/`python`: Python environment
- `cuda`/`CUDA`: GPU support
- `gcc`/`GCC`: C compiler
- `openmpi`/`OpenMPI`: MPI support

If your cluster uses different module names, modify the cluster setup tasks in `pixi.toml`.

### Batch Job Integration

You can use pixi environments in batch scripts:

**SLURM example:**
```bash
#!/bin/bash
#SBATCH --job-name=desc-job
#SBATCH --nodes=1
#SBATCH --gpus=1

# Load pixi and activate environment
eval "$(pixi shell -e cluster-gpu)"

# Run your DESC script
python your_desc_script.py
```

**PBS example:**
```bash
#!/bin/bash
#PBS -N desc-job
#PBS -l nodes=1:ppn=1:gpus=1

cd $PBS_O_WORKDIR
eval "$(pixi shell -e cluster-gpu)"
python your_desc_script.py
```

### Common Issues

1. **Module command not found**: Some clusters require `module` to be sourced:
   ```bash
   source /etc/profile.d/modules.sh  # or similar
   ```

2. **CUDA version mismatch**: Check your cluster's CUDA version and adjust the task accordingly.

3. **Python version conflicts**: Ensure the cluster's Python version matches the pixi environment (>=3.10, <=3.13).

## Customization

### Adding New Cluster Configurations

To add a new cluster configuration, add a task to `pixi.toml`:

```toml
setup-mycluster = """
if command -v module &> /dev/null; then
  module load my-python-module
  module load my-cuda-module
  # ... other cluster-specific modules
fi
"""
```

### Modifying Dependencies

- Core dependencies are in the main `[dependencies]` section
- Feature-specific dependencies are in `[feature.*.dependencies]` sections
- JAX and DESC-specific packages are installed via pip tasks

## Troubleshooting

### Check Environment
```bash
# List available environments
pixi info

# Check current environment
pixi shell --help

# List installed packages
pixi list
```

### Debug Installation
```bash
# Verbose pip installation
pip install -v -e .

# Check JAX installation
python -c "import jax; print(jax.devices())"

# Check DESC installation
python -c "from desc import __version__; print(__version__)"
```

For more help, see the [DESC documentation](https://desc-docs.readthedocs.io/) or [open an issue](https://github.com/PlasmaControl/DESC/issues).
