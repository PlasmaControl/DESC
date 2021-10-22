from warnings import WarningMessage
import pytest
import subprocess
import os
import h5py
import numpy as np
import time
from desc.__main__ import main

from desc.grid import Grid, LinearGrid, ConcentricGrid
from desc.basis import (
    PowerSeries,
    FourierSeries,
    DoubleFourierSeries,
    ZernikePolynomial,
    FourierZernikeBasis,
)
from desc.transform import Transform


@pytest.mark.benchmark(
    min_rounds=1, max_time=50, disable_gc=False, warmup=True, warmup_iterations=50
)
def test_build_transform_fft_lowres(benchmark):
    """Tests how long it takes to build a transform (after it has already been compiled) for lowres"""

    def build():
        L = 5
        M = 5
        N = 5
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark(build)


@pytest.mark.benchmark(min_rounds=1, max_time=100, disable_gc=False, warmup=True)
def test_build_transform_fft_midres(benchmark):
    """Tests how long it takes to build a transform (after it has already been compiled) for midres"""

    def build():
        L = 15
        M = 15
        N = 15
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark.pedantic(build, iterations=1, warmup_rounds=1, rounds=50)


@pytest.mark.benchmark(min_rounds=1, max_time=100, disable_gc=False, warmup=True)
def test_build_transform_fft_highres(benchmark):
    """Tests how long it takes to build a transform (after it has already been compiled) for highres"""

    def build():
        L = 25
        M = 25
        N = 25
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark.pedantic(build, iterations=1, warmup_rounds=1, rounds=25)
