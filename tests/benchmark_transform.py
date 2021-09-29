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
    min_rounds=1, max_time=50, disable_gc=False, warmup=False, warmup_iterations=50
)
def test_build_transform_fft_lowres(benchmark):
    def build():
        L = 5
        M = 5
        N = 5
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark(build)


@pytest.mark.benchmark(min_rounds=1, max_time=100, disable_gc=False, warmup=False)
def test_build_transform_fft_midres(benchmark):
    def build():
        L = 15
        M = 15
        N = 15
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark(build)


@pytest.mark.benchmark(min_rounds=1, max_time=100, disable_gc=False, warmup=False)
def test_build_transform_fft_highres(benchmark):
    def build():
        L = 25
        M = 25
        N = 25
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark(build)
