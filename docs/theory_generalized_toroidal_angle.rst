======================================
Generalized Toroidal Angle (``omega``)
======================================

DESC solves in the computational coordinates :math:`(\rho, \theta, \zeta)`.
:math:`\zeta` need not be the cylindrical laboratory angle :math:`\phi`: a
periodic toroidal stream function :math:`\omega` relates the two by

.. math::

   \phi(\rho, \theta, \zeta) = \zeta + \omega(\rho, \theta, \zeta),
   \qquad
   \mathbf{x} = \begin{pmatrix} R \cos\phi \\ R \sin\phi \\ Z \end{pmatrix},

so that

.. math::

   \phi_\rho = \omega_\rho, \qquad
   \phi_\theta = \omega_\theta, \qquad
   \phi_\zeta = 1 + \omega_\zeta .

:math:`\omega \equiv 0` recovers :math:`\phi = \zeta` and is the default for
every object, input file, and previously saved ``.h5`` file: no extra degrees
of freedom are created and no extra work is done.

Spectral representation
=======================

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - attribute
     - description
   * - ``eq.W_lmn``, ``eq.W_basis``
     - Fourier–Zernike coefficients and basis for :math:`\omega` in the volume
   * - ``eq.Lz``, ``eq.Mz``, ``eq.Nz``
     - independent radial/poloidal/toroidal resolution of that basis
   * - ``eq.Wb_lmn``
     - boundary coefficients (``surface.W_lmn``)
   * - ``eq.Wa_n``
     - axis coefficients (``axis.W_n``)

**Symmetry.** Under :math:`(\theta,\zeta) \to (-\theta,-\zeta)` both
:math:`\phi` and :math:`\zeta` are odd, so :math:`\omega` is odd and uses a
``sin`` parity basis, like :math:`Z` and :math:`\lambda`.

**Gauge.** :math:`\zeta \to \zeta + c(\rho)` with :math:`\omega \to \omega -
c(\rho)` leaves :math:`\phi` unchanged, so the :math:`(m=0,n=0)` content of
:math:`\omega` is pure gauge and is removed by ``FixOmegaGauge``. The ``sin``
basis has no :math:`(0,0)` modes, so under stellarator symmetry the gauge is
already fixed.

**On axis.** Every :math:`m \neq 0` mode vanishes at :math:`\rho = 0`, so
:math:`\omega(0,\theta,\zeta)` depends on :math:`\zeta` alone and the axis
remains a well defined curve. No extra regularity condition is needed.

Fitting a surface in generalized coordinates
============================================

Pass the toroidal coordinate you want as ``zeta``; DESC fits
:math:`\omega = \phi - \zeta` alongside :math:`R` and :math:`Z`::

    from desc.geometry import FourierRZToroidalSurface

    surf = FourierRZToroidalSurface.from_values(
        coords,        # (R, phi, Z), or (X, Y, Z) with basis="xyz"
        theta,         # the poloidal label of your samples
        zeta=phi_B,    # the desired computational toroidal coordinate
        M=16, N=20,    # R, Z resolution
        Mz=16, Nz=20,  # omega resolution (defaults to M, N)
        sym=False,
    )

Omitting ``zeta`` keeps ``zeta = phi`` and creates no :math:`\omega` modes.
:math:`\omega` is fitted as ``arctan2(sin(phi - zeta), cos(phi - zeta))``, which
is immune to :math:`2\pi` branch cuts in either input angle; the unit toroidal
winding is carried by the explicit :math:`\zeta` term.

.. note::

   Whatever array you pass as ``theta`` *becomes* the surface's poloidal
   coordinate. Two fits of the same physical surface using different poloidal
   labels have different parameterizations, so compare physical positions, not
   coefficient arrays.

To build an equilibrium on a generalized boundary, give it :math:`\omega`
resolution::

    eq = Equilibrium(L=8, M=8, N=8, surface=surf, Lz=0, Mz=0, Nz=4)

Map validity
============

An invertible chart requires :math:`\phi_\zeta = 1 + \omega_\zeta > 0`
everywhere. ``from_values`` checks this automatically; otherwise::

    surf.check_toroidal_map()   # returns min(1 + omega_zeta)
                                # raises if <= 0, warns if below tol

Constraints and solving
=======================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - constraint
     - purpose
   * - ``FixBoundaryW``
     - fix the boundary :math:`\omega` coefficients
   * - ``FixAxisW``
     - fix the axis :math:`\omega` coefficients
   * - ``BoundaryWSelfConsistency``
     - tie ``W_lmn`` at the boundary to ``Wb_lmn`` (added automatically)
   * - ``AxisWSelfConsistency``
     - tie ``W_lmn`` at :math:`\rho=0` to ``Wa_n`` (added automatically)
   * - ``FixOmegaGauge``
     - remove the :math:`\zeta \to \zeta + c(\rho)` gauge freedom
   * - ``FixOmegaInterior``
     - hold interior :math:`\omega` fixed (the conservative default)
   * - ``FixZetaSFL``
     - force :math:`\omega \equiv 0`, recovering the cylindrical angle

Interior :math:`\omega` is a coordinate choice, not physics: force balance is
degenerate along it. The default fixed-boundary problem therefore lets the
boundary supply the toroidal parameterization and keeps interior :math:`\omega`
fixed; freeing it is opt-in and requires regularization. ``FixOmegaInterior``
and ``FixOmegaGauge`` are added only when the equilibrium has :math:`\omega`
degrees of freedom, and the remaining constraints reduce to no-ops when it does
not, so ``omega = 0`` workflows are unaffected.

Current limitations
===================

* ``A(z)``, ``A``, ``A(r)``, ``a``, ``R0/a``, ``perimeter(z)`` and
  ``a_major/a_minor`` are computed on constant-:math:`\zeta` cross-sections.
  When :math:`\omega \neq 0` that is not a planar constant-:math:`\phi`
  section, so these become approximations.
* Plotting routines accept :math:`\omega \neq 0` equilibria, but a panel
  labelled "constant :math:`\phi`" is really constant :math:`\zeta` unless you
  supply a grid mapped with
  ``map_coordinates(..., inbasis=("rho","theta","phi"))``.
* ``map_coordinates`` is correct with :math:`\omega \neq 0` but falls back to a
  general Newton solve, which is slower than the specialized
  :math:`\omega = 0` paths.
* ``ZernikeRZToroidalSection`` is a constant-:math:`\zeta` object and always
  has :math:`\omega \equiv 0`.
* ``VMECIO.save`` requires :math:`\omega = 0` and raises otherwise.
* ``constant_offset_surface`` returns a surface carrying the base surface's
  :math:`\omega`, so the offset is measured normal to the base surface but the
  two share a toroidal chart; it does not re-fit :math:`\omega` for the offset
  geometry.
