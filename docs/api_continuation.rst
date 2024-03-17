======================
Solving and Perturbing
======================

Continuation
************
``desc.continuation`` contains the methods used for solving equilibrium problems.
``solve_continuation_automatic`` is usually the easiest method, users desiring more
control over the process can also use ``solve_continuation``.

.. autosummary::
    :toctree: _api/continuation
    :recursive:

    desc.continuation.solve_continuation_automatic
    desc.continuation.solve_continuation


Perturbations
*************
``desc.perturbations.perturb`` is used inside of the continuation methods but can
also be used alone to perform sensitivity analysis or perform parameter scans.
``optimal_perturb`` is effectively a single step of a constrained optimization solver.

.. autosummary::
    :toctree: _api/perturbations
    :recursive:

    desc.perturbations.perturb
    desc.perturbations.optimal_perturb
    desc.perturbations.get_deltas
