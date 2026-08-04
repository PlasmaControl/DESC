======================================
Generalized Toroidal Angle (``omega``)
======================================

DESC solves in the computational coordinates :math:`(\rho, \theta, \zeta)`.
Traditionally the toroidal coordinate :math:`\zeta` is *defined* to be the
cylindrical laboratory angle :math:`\phi`. That restriction can be lifted: DESC
supports a periodic **toroidal stream function** :math:`\omega` such that

.. math::

   \phi(\rho, \theta, \zeta) = \zeta + \omega(\rho, \theta, \zeta),

so the physical position of a point is

.. math::

   \mathbf{x}(\rho,\theta,\zeta) =
   \begin{pmatrix} R \cos(\zeta + \omega) \\
                   R \sin(\zeta + \omega) \\
                   Z \end{pmatrix}.

Setting :math:`\omega \equiv 0` recovers :math:`\phi = \zeta` exactly, and that
is the **default** for every object, every input file, and every previously
saved ``.h5`` file. If you never ask for a generalized angle, nothing changes:
no extra degrees of freedom are created and no extra work is done.

``zeta`` versus ``phi``
=======================

These are genuinely different quantities once :math:`\omega \neq 0`, and DESC
keeps them distinct in the data index:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - name
     - meaning
   * - ``zeta``
     - the computational toroidal coordinate; the third argument of every
       spectral basis and every ``Grid``
   * - ``phi``
     - the cylindrical laboratory angle, ``phi = zeta + omega``
   * - ``theta``
     - the computational poloidal angle
   * - ``theta_PEST``
     - straight field line poloidal angle at constant :math:`\phi`,
       ``theta + lambda``
   * - ``theta_B``, ``phi_B``
     - Boozer angles. In general :math:`\phi_B \neq \phi`
   * - ``X``, ``Y``, ``Z``
     - the physical Cartesian position

A constant-:math:`\zeta` surface is a constant-:math:`\phi` plane **only** when
:math:`\omega = 0`. Quantities that are physically defined on a planar
cross-section are listed under `Limitations`_ below.

The meaning of ``omega``
========================

:math:`\omega` is the angular displacement between the computational toroidal
coordinate and the laboratory angle. It is stored spectrally:

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

Derivatives follow immediately from :math:`\phi = \zeta + \omega`:

.. math::

   \phi_\rho = \omega_\rho, \qquad
   \phi_\theta = \omega_\theta, \qquad
   \phi_\zeta = 1 + \omega_\zeta,

and likewise for all higher derivatives. Every geometric quantity in DESC
(covariant and contravariant basis vectors, the Jacobian, metric tensors,
curvature, normals, the magnetic field, force balance) is built from these, so
they are all generalized automatically.

**Symmetry.** Under stellarator symmetry :math:`(\theta,\zeta) \to
(-\theta,-\zeta)` both :math:`\phi` and :math:`\zeta` are odd, so
:math:`\omega` is odd as well: it uses a ``sin`` parity basis, like :math:`Z`
and :math:`\lambda`.

**Gauge.** For a fixed physical field the computational angle may be shifted
per flux surface, :math:`\zeta \to \zeta + c(\rho)` with :math:`\omega \to
\omega - c(\rho)`, leaving :math:`\phi` unchanged. The :math:`(m=0,n=0)`
content of :math:`\omega` is therefore pure gauge, removed by
``FixOmegaGauge``. Under stellarator symmetry the ``sin`` basis contains no
:math:`(0,0)` modes and the gauge is fixed automatically.

**On axis.** Every :math:`m \neq 0` Fourier–Zernike mode vanishes at
:math:`\rho = 0`, so :math:`\omega(0,\theta,\zeta)` is automatically a function
of :math:`\zeta` alone and the magnetic axis remains a well defined curve. No
additional regularity condition is required.

Providing Boozer or other generalized toroidal coordinates
==========================================================

The usual entry point is fitting a surface from sampled points. Pass the
toroidal coordinate you want as ``zeta``; DESC fits
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

Omitting ``zeta`` keeps the classical behavior: ``zeta = phi``, no
:math:`\omega` modes are created at all.

.. note::

   The surface attaches no intrinsic meaning to its poloidal angle. **Whatever
   array you pass as** ``theta`` **becomes** the surface's poloidal coordinate.
   Passing :math:`\theta_{\mathrm{PEST}}` samples gives a surface parameterized
   by :math:`\theta_{\mathrm{PEST}}`; passing :math:`\theta_B` gives one
   parameterized by :math:`\theta_B`. Two such fits of the *same* physical
   surface have different parameterizations, so their coefficient arrays are
   not comparable — compare physical positions instead.

:math:`\omega` is computed as the continuous periodic displacement
``arctan2(sin(phi - zeta), cos(phi - zeta))``, which is immune to
:math:`2\pi` branch cuts in either input angle. The unit toroidal winding is
carried entirely by the explicit :math:`\zeta` term, so the fitted map always
has the correct degree-one winding.

To build an equilibrium on a generalized boundary, give it :math:`\omega`
resolution::

    eq = Equilibrium(L=8, M=8, N=8, surface=surf, Lz=0, Mz=0, Nz=4)

Map validity requirements
=========================

:math:`(\rho,\theta,\zeta) \mapsto (R,\phi,Z)` is a valid toroidal chart only
if the toroidal winding of :math:`\zeta` matches that of :math:`\phi` along
every coordinate line, that is

.. math::

   \frac{\partial \phi}{\partial \zeta}
   = 1 + \frac{\partial \omega}{\partial \zeta} > 0
   \qquad \text{everywhere.}

If this vanishes or changes sign, constant-:math:`\zeta` surfaces fold over in
:math:`\phi` and the map is not invertible. ``from_values`` checks this after
fitting, and you can check any surface yourself::

    surf.check_toroidal_map()   # returns min(1 + omega_zeta); raises if <= 0

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
boundary supply the toroidal parameterization and keeps interior
:math:`\omega` determined. Freeing it is opt-in and requires you to think about
regularization. All of these constraints are added **only** when the
equilibrium actually has :math:`\omega` degrees of freedom, so ``omega = 0``
workflows are completely unaffected.

.. _Limitations:

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
* ``VMECIO.save`` and ``FourierRZToroidalSurface.constant_offset_surface``
  require :math:`\omega = 0` and raise otherwise.
* Continuation keeps :math:`\omega` fixed.
