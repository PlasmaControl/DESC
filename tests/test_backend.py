"""Tests for backend functions."""

import shutil

import numpy as np
import pytest

from desc.backend import (
    _lstsq,
    jax,
    jnp,
    put,
    qr,
    qr_multiply,
    root,
    root_scalar,
    sign,
    solve_triangular,
    vmap,
)


@pytest.mark.unit
def test_put():
    """Test put function as replacement for fancy array indexing."""
    a = np.array([0, 0, 0])
    b = np.array([1, 2, 3])
    a = put(a, np.array([0, 1, 2]), np.array([1, 2, 3]))
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_sign():
    """Test modified sign function to return +1 for x=0."""
    assert sign(4) == 1
    assert sign(0) == 1
    assert sign(-10.3) == -1


@pytest.mark.unit
def test_vmap():
    """Test lax numpy implementation of Python's map function."""
    a = np.arange(6)
    inputs = np.stack([a, a[::-1], -a])

    def f(x):
        return x[: x.size // 2] ** 3

    outputs = np.array([[0, 1, 8], [125, 64, 27], [0, -1, -8]])
    np.testing.assert_allclose(vmap(f)(inputs), outputs)
    np.testing.assert_allclose(vmap(f, out_axes=1)(inputs), outputs.T)


@pytest.mark.unit
def test_root():
    """Test root and its derivative works properly."""

    def fun(x, a):
        return a * x - 1

    def find_root(a):
        x0 = jnp.zeros_like(a)
        xk = root(fun, x0, args=(a,))
        return xk

    def find_root_full(a):
        x0 = jnp.zeros_like(a)
        xk, aux = root(fun, x0, args=(a,), full_output=True)
        return xk, aux

    a = 2 * jnp.ones(10)
    x = find_root(a)
    x_full, _ = find_root_full(a)

    exact = 1 / a
    np.testing.assert_allclose(x, exact)
    np.testing.assert_allclose(x_full, exact)

    J = jax.jit(jax.jacfwd(find_root))(a)
    J_rev = jax.jit(jax.jacrev(find_root))(a)
    J_full, _ = jax.jit(jax.jacfwd(find_root_full))(a)
    J_full_rev, _ = jax.jit(jax.jacrev(find_root_full))(a)
    J_exact = jnp.diag(-1 / a**2)

    np.testing.assert_allclose(J, J_exact)
    np.testing.assert_allclose(J_full, J_exact)
    np.testing.assert_allclose(J_rev, J_exact)
    np.testing.assert_allclose(J_full_rev, J_exact)


@pytest.mark.unit
def test_root_scalar():
    """Test root_scalar and its derivative works properly."""

    def fun(x, a):
        return a * x - 1

    def find_root(a):
        x0 = 0.0
        xk = root_scalar(fun, x0, args=(a,))
        return xk

    def find_root_full(a):
        x0 = 0.0
        xk, aux = root_scalar(fun, x0, args=(a,), full_output=True)
        return xk, aux

    a = 2.0
    x = find_root(a)
    x_full, _ = find_root_full(a)

    exact = 1 / a
    np.testing.assert_allclose(x, exact)
    np.testing.assert_allclose(x_full, exact)

    J = jax.jit(jax.jacfwd(find_root))(a)
    J_rev = jax.jit(jax.jacrev(find_root))(a)
    J_full, _ = jax.jit(jax.jacfwd(find_root_full))(a)
    J_full_rev, _ = jax.jit(jax.jacrev(find_root_full))(a)
    J_exact = -1 / a**2

    np.testing.assert_allclose(J, J_exact)
    np.testing.assert_allclose(J_full, J_exact)
    np.testing.assert_allclose(J_rev, J_exact)
    np.testing.assert_allclose(J_full_rev, J_exact)


@pytest.mark.unit
def test_lstsq():
    """Test cholesky factorization of least squares solution."""
    rng = np.random.default_rng(seed=0)

    # tall
    A = rng.standard_normal((10, 5))
    b = rng.standard_normal(10)
    np.testing.assert_allclose(
        _lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-6
    )
    # wide
    A = rng.standard_normal((5, 10))
    b = rng.standard_normal(5)
    np.testing.assert_allclose(
        _lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-6
    )
    # square
    A = rng.standard_normal((5, 5))
    b = rng.standard_normal(5)
    np.testing.assert_allclose(
        _lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-6
    )
    # scalar
    A = rng.standard_normal((1, 5))
    b = rng.standard_normal(1)
    np.testing.assert_allclose(
        _lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-6
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "m, n",
    [
        (100, 20),  # tall
        (20, 100),  # wide
        (50, 50),  # square
        (600, 260),  # tall, and has more than nb columns
    ],
)
@pytest.mark.parametrize("cond", [None, 1e8])
def test_qr_multiply(m, n, cond):
    """Test qr_multiply matches forming Q explicitly."""

    def _qr_multiply_ref(a, c, mode="right"):
        """Reference qr_multiply that forms Q explicitly."""
        Q, R = qr(a, mode="economic")
        if mode == "right":
            cq = Q.T @ c if c.ndim == 1 else c @ Q
        else:
            cq = Q @ c
        return cq, R

    rng = np.random.default_rng(seed=0)
    k = min(m, n)

    if cond is None:
        # gaussian matrices are usually well conditioned
        A = rng.standard_normal((m, n))
    else:
        # create some ill conditioned matrix using reverse SVD
        # this is still full rank
        U = np.linalg.qr(rng.standard_normal((m, k)))[0]
        V = np.linalg.qr(rng.standard_normal((n, k)))[0]
        A = (U * np.logspace(0, -np.log10(cond), k)) @ V.T
    b = rng.standard_normal(m)
    print(
        f"Running {m=} {n=} {cond=}, actual condition number is {np.linalg.cond(A):.3e}"
    )

    # mode="right" with 1D c is Q.T@b
    Qtb, R = qr_multiply(A, b, mode="right")
    Qtb_ref, R_ref = _qr_multiply_ref(A, b, mode="right")
    assert R.shape == (k, n)
    np.testing.assert_allclose(R, R_ref, rtol=1e-12, atol=1e-12 * np.abs(A).max())
    np.testing.assert_allclose(Qtb, Qtb_ref, rtol=1e-10, atol=1e-10)

    # mode="right" with 2D c is c@Q
    C = rng.standard_normal((3, m))
    CQ, _ = qr_multiply(A, C, mode="right")
    np.testing.assert_allclose(CQ, _qr_multiply_ref(A, C, "right")[0], atol=1e-10)

    # mode="left" is Q@c, with c=I recovers Q
    Q, _ = qr_multiply(A, np.eye(k), mode="left")
    np.testing.assert_allclose(Q, _qr_multiply_ref(A, np.eye(k), "left")[0], atol=1e-10)
    np.testing.assert_allclose(Q.T @ Q, np.eye(k), atol=1e-10)
    np.testing.assert_allclose(Q @ R, A, atol=1e-10 * np.abs(A).max())
    y = rng.standard_normal(k)
    Qy, _ = qr_multiply(A, y, mode="left")
    np.testing.assert_allclose(Qy, _qr_multiply_ref(A, y, "left")[0], atol=1e-10)

    # solve A@x = b
    if m >= n:
        x = solve_triangular(R, Qtb)
        x_ref = solve_triangular(R_ref, Qtb_ref)
    else:
        # for wide A, use the QR of A.T
        Q1, R1 = qr_multiply(A.T, np.eye(k), mode="left")
        Q1_ref, R1_ref = _qr_multiply_ref(A.T, np.eye(k), mode="left")
        x = Q1 @ solve_triangular(R1.T, b, lower=True)
        x_ref = Q1_ref @ solve_triangular(R1_ref.T, b, lower=True)
    x_np = np.linalg.lstsq(A, b, rcond=None)[0]
    np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-8 * np.abs(x_np).max())
    np.testing.assert_allclose(x, x_np, rtol=1e-8, atol=1e-8 * np.abs(x_np).max())


@pytest.mark.unit
def test_cgroup_available_memory(tmp_path):
    """The tightest cgroup memory limit of the process and its parents counts."""
    from desc import _cgroup_available_memory

    gib = 1024**3
    # cgroup v2: a job inside a user slice, both limited, page cache reclaimable
    root = tmp_path / "cgroup"
    job = root / "user" / "job"
    job.mkdir(parents=True)
    (root / "user" / "memory.max").write_text(f"{8 * gib}\n")
    (root / "user" / "memory.current").write_text(f"{3 * gib}\n")
    (job / "memory.max").write_text(f"{4 * gib}\n")
    (job / "memory.current").write_text(f"{3 * gib}\n")
    (job / "memory.stat").write_text(f"anon {2 * gib}\ninactive_file {gib}\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/user/job\n")
    assert _cgroup_available_memory(str(proc), str(root)) == 2 * gib
    # no limits
    (job / "memory.max").write_text("max\n")
    (root / "user" / "memory.max").write_text("max\n")
    assert _cgroup_available_memory(str(proc), str(root)) == float("inf")
    # cgroup v1 memory controller, its "no limit" a number close to 2**63
    v1 = root / "memory" / "slurm" / "job_1"
    v1.mkdir(parents=True)
    (v1 / "memory.limit_in_bytes").write_text(f"{2 * gib}\n")
    (v1 / "memory.usage_in_bytes").write_text(f"{gib}\n")
    (root / "memory" / "memory.limit_in_bytes").write_text(f"{2**63 - 4096}\n")
    (root / "memory" / "memory.usage_in_bytes").write_text(f"{gib}\n")
    proc.write_text("12:cpuset:/slurm/job_1\n4:memory:/slurm/job_1\n")
    assert _cgroup_available_memory(str(proc), str(root)) == gib
    # cgroup v1 with the memory controller mounted together with another one
    v1 = root / "cpu,memory" / "slurm" / "job_1"
    v1.mkdir(parents=True)
    (v1 / "memory.limit_in_bytes").write_text(f"{3 * gib}\n")
    (v1 / "memory.usage_in_bytes").write_text(f"{gib}\n")
    shutil.rmtree(root / "memory")
    proc.write_text("4:cpu,memory:/slurm/job_1\n")
    assert _cgroup_available_memory(str(proc), str(root)) == 2 * gib
    # not in a cgroup at all
    assert _cgroup_available_memory(str(tmp_path / "none"), str(root)) == float("inf")


@pytest.mark.unit
def test_set_device_cpu_respects_cgroup_limit(monkeypatch):
    """On CPU, the available memory is no more than the cgroup limit leaves."""
    import desc

    if desc.config["kind"] != "cpu":
        pytest.skip("only on CPU")
    for key in ["kind", "device", "avail_mem"]:
        monkeypatch.setitem(desc.config, key, desc.config[key])
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(desc, "_cgroup_available_memory", lambda: 0.5 * 1024**3)
    desc.set_device("cpu")
    assert desc.config["avail_mem"] == 0.5
    monkeypatch.setattr(desc, "_cgroup_available_memory", lambda: float("inf"))
    desc.set_device("cpu")
    assert 0 < desc.config["avail_mem"] < float("inf")


class _FakeGPU:
    """A device that reports the memory statistics it is given."""

    platform = "gpu"

    def __init__(self, stats):
        self.stats = stats

    def memory_stats(self):
        return self.stats


@pytest.mark.unit
def test_jax_available_memory(monkeypatch):
    """On a GPU, the memory left in JAX's pool counts, not all free memory."""
    from desc.backend import _jax_available_memory, desc_config

    gib = 1024**3
    monkeypatch.setitem(desc_config, "avail_mem", 15.0)
    # JAX's pool: 75% of a 16 GiB GPU, 1 GiB of it in use
    gpu = _FakeGPU({"bytes_limit": 12 * gib, "bytes_in_use": gib})
    assert _jax_available_memory(gpu) == 11
    # a pool larger than the free memory
    gpu = _FakeGPU({"bytes_limit": 32 * gib, "bytes_in_use": 0})
    assert _jax_available_memory(gpu) == 15
    # no pool (eg XLA_PYTHON_CLIENT_ALLOCATOR=platform)
    assert _jax_available_memory(_FakeGPU(None)) == 15
    # CPU: the available memory found by set_device
    assert _jax_available_memory(jax.devices("cpu")[0]) == 15
