.. _input_file:

==========
Input File
==========

The following is an example DESC input file, which contains all of the available input arguments.
This example is only intended to demonstrate the input file format, and may not necessarily converge well.
More realistic input examples are included in the repository.
DESC can also accept VMEC input files, which are converted to DESC inputs as explained below (however not all desc solver options have VMEC analogs, see below).

.. code-block:: text
   :linenos:

   ! this is a comment
   # this is also a comment

   # global parameters
   sym = 1
   NFP = 5
   Psi = 1.0

   # spectral resolution
   L_rad  =  4:4:24
   M_pol  =  6:2:10, 10; 11x2 12
   N_tor  =  0  1;  2x2,  3x2;  4
   L_grid =  8:4:32
   M_grid =  9:3:15, 16:1:18
   N_grid =  0  2  3,  3;  4  5  6

   # continuation parameters
   bdry_ratio = 0:0.5:1
   pres_ratio = 0x2, 0:0.5:1
   pert_order = 2

   # solver tolerances
   ftol = 1e-2
   xtol = 1e-6
   gtol = 1e-6
   maxiter = 100

   # solver methods
   objective         = force
   optimizer         = lsq-exact
   spectral_indexing = ansi
   node_pattern      = jacobi

   # pressure and rotational transform/current profiles
   l:   0   p =  1.80000000E+04   i =  1.0
   l:   2   p = -3.60000000E+04   i =  1.5
   l:   4   p =  1.80000000E+04

   # magnetic axis initial guess
   n:   0  R0 =  10  Z0 =  0.0

   # fixed-boundary surface shape
   m:   0   n:   0  R1 =  1.00000000E+01  Z1 =  0.00000000E+00
   m:   1   n:   0  R1 =  1.00000000E+00
   m:   1   n:   1  R1 =  3.00000000E-01
   m:  -1   n:  -1  R1 = -3.00000000E-01
   m:  -1   n:   0  Z1 =  1.00000000E+00
   m:   1   n:  -1  Z1 = -3.00000000E-01
   m:  -1   n:   1  Z1 = -3.00000000E-01

General Notes
*************

Both ``!`` and ``#`` are recognized to denote comments at the end of a line.
Whitespace is always ignored, except for newline characters.
Multiple inputs can be given on the same line of the input file, but a single input cannot span multiple lines.
None of the inputs are case-sensitive; for example ``M_pol``, ``M_POL``, and ``m_Pol`` are all the same.
All numerical values can be given in either decimal or exponential formats.

Global Parameters
*****************

.. code-block:: text

   sym = 1
   NFP = 5
   Psi = 1.0

- ``sym`` (bool): True (1) to assume stellarator symmetry, False (0) otherwise. Default = 0.
- ``NFP`` (int): Number of toroidal field periods, :math:`N_{FP}` (discrete toroidal symmetry). Default = 1.
- ``Psi`` (float): The toroidal magnetic flux through the last closed flux surface, :math:`\psi_a` (Webers). Default = 1.0.

Spectral Resolution
*******************

.. code-block:: text

   L_rad  =  4:4:24
   M_pol  =  6:2:10, 10; 11x2 12
   N_tor  =  0  1;  2x2,  3x2;  4
   L_grid =  8:4:32
   M_grid =  9:3:15, 16:1:18
   N_grid =  0  2  3,  3;  4  5  6

- ``L_rad`` (int): Maximum radial mode number for the Fourier-Zernike basis, :math:`L`. Default = ``M_pol`` if ``spectral_indexing = ANSI``, or ``2*M_pol`` if ``spectral_indexing = Fringe``. For more information see `Basis functions and collocation nodes`_.
- ``M_pol`` (int): Maximum poloidal mode number for the Fourier-Zernike basis, :math:`M`. Required.
- ``N_tor`` (int): Maximum toroidal mode number for the Fourier-Zernike basis, :math:`N`. Default = 0.
- ``L_grid`` (int): Radial resolution of nodes in collocation grid. Default = ``M_grid`` if ``spectral_indexing = ANSI``, or ``2*M_grid`` if ``spectral_indexing = Fringe``.
- ``M_grid`` (int): Poloidal resolution of nodes in collocation grid. Default = ``2*M_pol``.
- ``N_grid`` (int): Toroidal resolution of nodes in collocation grid. Default = ``2*N_tor``.

When ``M_grid = M_pol`` the number of collocation nodes in each toroidal cross-section is equal to the number of Zernike polynomial in the basis set.
When ``N_grid = N_tor`` the number of nodes with unique toroidal angles is equal to the number of terms in the toroidal Fourier series.
Convergence is typically superior when the number of nodes exceeds the number of spectral coefficients, but this adds compuational cost.

These arguments can be passed as arrays, where each element denotes the value to use at that iteration.
Array elements are delimited by either a space `` ``, comma ``,``, or semicolon ``;``.
Arrays can also be created using the shorthand notation ``start:interval:end`` and ``(value)x(repititions)``.
For example, the input line for ``M_pol`` shown above is equivalent to ``M_pol = 6, 8, 10, 10, 11, 11, 12``.
In this example there will be 7 iterations; any array with fewer than 7 elements will use its final value for the remaining iterations.

Continuation Parameters
***********************

.. code-block:: text

   pres_ratio = 0:0.5:1
   bdry_ratio = 0x2, 0:0.5:1
   pert_order = 2

- ``pres_ratio`` (float): Multiplier on the pressure profile. Default = 1.0.
- ``bdry_ratio`` (float): Multiplier on the 3D boundary modes. Default = 1.0.
- ``pert_order`` (int): Order of the perturbation approximation: 0 = no perturbation, 1 = linear, 2 = quadratic. Default = 1.

When both ``pres_ratio = 1`` and ``pres_ratio = 1``, the equilibrium is solved using the exact boundary modes and pressure profile as input.
``pres_ratio = 0`` assumes a vacuum pressure profile, and ``bdry_ratio = 0`` ignores all of the non-axisymmetric boundary modes (reducing the input to a tokamak).

These arguments are also passed as arrays for each iteration, with the same notation as the other continuation parameters.
This example will start by solving a vacuum tokamak, then perturb the pressure profile to solve a finite-beta tokamak, and finally perturb the boundary to solve the finite-beta stellarator.
If only one value is given, as with ``pert_order`` in this example, that value will be used for all iterations.

If ``pres_ratio`` and ``bdry_ratio`` are not present in the input file, and only 1 set of resolutions are specified,
an adaptive automatic continuation method will be used.

Solver Tolerances
*****************

.. code-block:: text

   ftol = 1e-2
   xtol = 1e-6
   gtol = 1e-6
   nfev = 100

- ``ftol`` (float): Solver stopping tolerance on the relative norm of dF. Default = 1e-2.
- ``xtol`` (float): Solver stopping tolerance on the relative norm of dx. Default = 1e-6.
- ``gtol`` (float): Solver stopping tolerance on the norm of the gradient. Default = 1e-8.
- ``maxiter`` (int): Maximum number of optimizer iterations. Default = 100.

These arguments are also passed as arrays for each iteration, with the same notation as the other continuation parameters.
In this example, the same values are being used for all 7 iterations.

Solver Methods
**************

.. code-block:: text

   objective         = force
   optimizer         = lsq-exact
   spectral_indexing = fringe
   node_pattern      = jacobi

- ``objective`` (string): Form of equations to use for solving the equilibrium. Options are ``force`` (Default), ``forces``, ``energy``, or ``vacuum``.
- ``optimizer`` (string): Type of optimizer to use. Default = ``lsq-exact``. For more details and options see :py:class:`desc.optimize.Optimizer`.
- ``spectral_indexing`` (string): Zernike polynomial index ordering. Options are ``ANSI`` or ``Fringe`` (Default). For more information see `Basis functions and collocation nodes`_.
- ``node_pattern`` (string): Pattern of collocation nodes. Options are ``jacobi`` (Default), ``cheb1``, ``cheb2`` or ``quad``. For more information see `Basis functions and collocation nodes`_.

The ``objective`` option ``force`` minimizes the equilibrium force balance errors in units of Newtons, while the ``energy`` option minimizes the total plasma energy in units of Joules.

Pressure & Iota/Current Profiles
********************************

.. code-block:: text

   iota = 1
   l:   0   p =  1.80000000E+04   i =  1.0
   l:   2   p = -3.60000000E+04   i =  1.5
   l:   4   p =  1.80000000E+04

- ``l`` (int): Radial polynomial order.
- ``p`` (float): Pressure profile coefficients :math:`p_{l}`.
- ``i`` (float): Rotational transform coefficients :math:`\iota_{l}`.
- ``c`` (float): Toroidal current coefficients :math:`c_{l}`.

The profiles are given as a power series in the flux surface label :math:`\rho \equiv \sqrt{\psi / \psi_a}` as follows:

.. math::
   \begin{aligned}
   p(\rho) &= \sum p_{l} \rho^{l} \\
   \iota(\rho) &= \sum \iota_{l} \rho^{l} \\
   \frac{2\pi}{\mu_0} I(\rho) &= \sum c_{l} \rho^{l} \\.
   \end{aligned}

The coefficients :math:`p_{l}` are specified by the input variables ``p`` in Pascals.
The coefficients :math:`\iota_{l}` are specified by the input variables ``i``.
The coefficients :math:`c_{l}` are specified by the input variables ``c`` in Amperes.
Either the rotational transform or toroidal current profiles can be specified, but not both.
The radial exponent :math:`l` is given by ``l``, which must be on the same input line as the coefficients.
The profiles given in the example are:

.. math::
   \begin{aligned}
   p(\rho) &= 1.8\times10^4 (1-\rho^2)^2 \\
   \iota(\rho) &= 1 + 1.5 \rho^2.
   \end{aligned}

If no profile inputs are given, it is assumed that they are :math:`p(\rho) = 0` and :math:`\frac{2\pi}{\mu_0} I(\rho) = 0`.
Also, note that the rotational transform given is technically assumed to be

.. math::
   \begin{aligned}
    \mbox{$\,\iota\!\!$- }= \iota / 2\pi
    \end{aligned}

i.e. rational surfaces would be where the input rotational transform profile is equal to a rational number.

Magnetic Axis Initial Guess
***************************

.. code-block:: text

   n:   0  R0 =  10  Z0 =  0.0

- ``n`` (int): Toroidal mode number.
- ``R0`` (float): Fourier coefficient of the R coordinate of the magnetic axis. :math:`R^{0}_{n}`
- ``Z0`` (float): Fourier coefficient of the Z coordinate of the magnetic axis. :math:`Z^{0}_{n}`

An initial guess for the magnetic axis can be supplied in the form:

.. math::
   \begin{aligned}
   R_{0}(\phi) &= \sum_{n=-N}^{N} R^{0}_{n} \mathcal{F}_{n}(\phi) \\
   Z_{0}(\phi) &= \sum_{n=-N}^{N} Z^{0}_{n} \mathcal{F}_{n}(\phi) \\
   \mathcal{F}_{n}(\phi) &= \begin{cases}
   \cos(|n|N_{FP}\phi) &\text{for }n\ge0 \\
   \sin(|n|N_{FP}\phi) &\text{for }n<0. \\
   \end{cases}
   \end{aligned}

The coefficients :math:`R^{0}_{n}` and :math:`Z^{0}_{n}` are specified by the input variables ``R0`` and ``Z0``, respectively.
The Fourier mode number :math:`n` is given by ``n``, which must be on the same input line as the coefficients.

The magnetic axis initial guess is optional and only used if ``eq.surface.type = FourierRZToroidalSurface``.
If no initial guess is provided for the magnetic axis, then the :math:`m = 0` modes of the fixed-boundary surface shape input are used.

Fixed-Boundary Surface Shape
****************************

.. code-block:: text

   m:   0   n:   0  R1 =  1.00000000E+01  Z1 =  0.00000000E+00
   m:   1   n:   0  R1 =  1.00000000E+00
   m:   1   n:   1  R1 =  3.00000000E-01
   m:  -1   n:  -1  R1 = -3.00000000E-01
   m:  -1   n:   0  Z1 =  1.00000000E+00
   m:   1   n:  -1  Z1 = -3.00000000E-01
   m:  -1   n:   1  Z1 = -3.00000000E-01

- ``m`` (int): Poloidal mode number.
- ``n`` (int): Toroidal mode number. (Only used if ``eq.surface.type = FourierRZToroidalSurface``.)
- ``R1`` (float): Fourier coefficient of the R coordinate of the boundary surface. :math:`R^{1}_{mn}`
- ``Z1`` (float): Fourier coefficient of the Z coordinate of the boundary surface. :math:`Z^{1}_{mn}`

If ``eq.surface.type = FourierRZToroidalSurface``, the shape of the last closed flux surface is given as a double Fourier series of the form:

.. math::
   \begin{aligned}
   R_{1}(\theta,\phi) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} R^{1}_{mn} \mathcal{G}^{m}_{n}(\theta,\phi) \\
   Z_{1}(\theta,\phi) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} Z^{1}_{mn} \mathcal{G}^{m}_{n}(\theta,\phi) \\
   \mathcal{G}^{m}_{n}(\theta,\phi) &= \begin{cases}
   \cos(|m|\theta)\cos(|n|N_{FP}\phi) &\text{for }m\ge0, n\ge0 \\
   \cos(|m|\theta)\sin(|n|N_{FP}\phi) &\text{for }m\ge0, n<0 \\
   \sin(|m|\theta)\cos(|n|N_{FP}\phi) &\text{for }m<0, n\ge0 \\
   \sin(|m|\theta)\sin(|n|N_{FP}\phi) &\text{for }m<0, n<0.
   \end{cases}
   \end{aligned}

The coefficients :math:`R^{1}_{mn}` and :math:`Z^{1}_{mn}` are specified by the input variables ``R1`` and ``Z1``, respectively.
The spectral mode numbers :math:`l`, :math:`m`, and :math:`n` are given by ``l``, ``m``, and ``n``, respectively, which must be on the same input line as the coefficients.
The fixed-boundary surface shape is a required input.

The fixed-boundary surface shape given in this example is equivalent to (using Ptolemy’s identities):

.. math::
   \begin{aligned}
   R_{1}(\theta,\phi) &= 10 + \cos\theta + 0.3 \cos(\theta+19\phi) \\
   Z_{1}(\theta,\phi) &= \sin\theta - 0.3 \sin(\theta+19\phi).
   \end{aligned}

VMEC Inputs
***********

A VMEC input file can also be passed in place of a DESC input file.
DESC will detect if it is a VMEC input format and automatically generate an equivalent DESC input file.
The generated DESC input file will be stored at the same file path as the VMEC input file, but its name will have ``_desc`` appended to it.
The resulting input file will not contain any of the options that are specific to DESC, and therefore will depend on many default values.
This is a convenient tool for converting the profiles and boundary inputs to the DESC format, but the generated input file may not converge well with the default options for all equilibria.
It is recommended that the automatically generated DESC input file be manually edited to improve performance.
As an example, see the simple VMEC input file below titled ``input.HELIOTRON``:

.. code-block:: text

   &INDATA
   LFREEB =	F
   DELT =	0.9
   TCON0 =	2
   LASYM =	F
   NFP =	19
   NCURR =	0
   NZETA =	200
   NITER_ARRAY =	4000 8000 12000 16000 32000
   FTOL_ARRAY =	1e-8 1e-9 1e-10 1e-11 1e-12
   NSTEP =	250
   NVACSKIP =	6
   GAMMA =	0
   PHIEDGE =	1
   BLOAT =	1
   CURTOR =	0
   SPRES_PED =	1
   PRES_SCALE =	18000.0
   PMASS_TYPE =	"power_series"
   RAXIS =	10
   ZAXIS =	0
   AM =	1 -2 1
   AI =	1.0 1.5
   RBC(0,0) =	10.000000
   RBC(0,1) =	-1.000000
   RBC(-1,0) =	0.000000
   RBC(-1,1) =	-0.300000
   ZBS(0,0) =	0.000000
   ZBS(0,1) =	1.000000
   ZBS(-1,0) =	0.000000
   ZBS(-1,1) =	-0.300000
   MPOL =	6
   NTOR =	3
   NS_ARRAY =	16 32 64 128 256
   /
   &END

Upon running ``desc input.HELIOTRON`` from the command line, the DESC code will automatically convert the VMEC input into a DESC input file and run it.
The DESC input file will be this, titled ``input.HELIOTRON_desc``:

.. code-block:: text

   # This DESC input file was auto generated from a VMEC input file
   # For details on the various options see https://desc-docs.readthedocs.io/en/stable/input.html

   # global parameters
   sym = 1
   NFP =  19
   Psi = 1.00000000

   # spectral resolution
   L_rad = 6
   M_pol = 6
   N_tor = 3
   L_grid = 12
   M_grid = 12
   N_grid = 6

   # continuation parameters
   pert_order = 2.0

   # solver tolerances

   # solver methods
   optimizer = lsq-exact
   objective = force
   spectral_indexing = ansi
   node_pattern = jacobi

   # pressure and rotational transform/current profiles
   l:   0	p =   1.80000000E+04	i =   1.00000000E+00
   l:   2	p =  -3.60000000E+04	i =   1.50000000E+00
   l:   4	p =   1.80000000E+04	i =   0.00000000E+00

   # fixed-boundary surface shape
   l:   0	m:  -1	n:  -1	R1 =   3.00000000E-01	Z1 =   0.00000000E+00
   l:   0	m:  -1	n:   0	R1 =   0.00000000E+00	Z1 =   1.00000000E+00
   l:   0	m:  -1	n:   1	R1 =   0.00000000E+00	Z1 =  -3.00000000E-01
   l:   0	m:   0	n:  -1	R1 =   0.00000000E+00	Z1 =   0.00000000E+00
   l:   0	m:   0	n:   0	R1 =   1.00000000E+01	Z1 =   0.00000000E+00
   l:   0	m:   0	n:   1	R1 =   0.00000000E+00	Z1 =   0.00000000E+00
   l:   0	m:   1	n:  -1	R1 =   0.00000000E+00	Z1 =  -3.00000000E-01
   l:   0	m:   1	n:   0	R1 =  -1.00000000E+00	Z1 =   0.00000000E+00
   l:   0	m:   1	n:   1	R1 =  -3.00000000E-01	Z1 =   0.00000000E+00

   # magnetic axis initial guess
   n:   0	R0 =   1.00000000E+01	Z0 =   0.00000000E+00


You can see that the main elements of the input file are present here.
See the example DESC input files on the github repository to see typical choices of solver options for some common equilibria, as well as the `arxiv publication on the DESC perturbation and continuation methods <https://arxiv.org/abs/2203.15927>`_ .

Some general considerations

The continuation parameters ``pres_ratio`` and ``bdry_ratio`` are important for complex equilibria.
Setting these in arrays such as shown in the above section, such that first a vacuum tokamak is solved, then finite beta tokamak, and finally the non-axisymmetric modes are added, is recommended for best results for highly shaped stellarator equilibria.
Equally important are the spectral resolution parameters ``L_rad``, ``L_grid``, ``M_pol``, ``M_grid``, ``N_tor``, and ``N_grid``
Starting with a low spectral resolution, then increasing the number of modes in the basis is found to achieve faster results as compared to starting the equilibrium solve with the full desired resolution.

.. _Basis functions and collocation nodes: notebooks/basis_grid.ipynb
