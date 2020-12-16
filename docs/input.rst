.. _input_file:

==========
Input File
==========

The following is an example DESC input file, which containts all of the available input arguments. 
More input examples are included in the repository. 
DESC can also accept VMEC input files, which are converted to DESC inputs as explained below. 

.. code-block:: text
   :linenos:

   ! this is a comment
   # this is also a comment
   
   # global parameters
   stell_sym = 1
   NFP       = 19
   Psi_lcfs  = 1.00000000E+00
   
   # spectral resolution
   Mpol     =   6   8  10  10  11  11  12
   delta_lm =   0   4   8  12  16  20  24
   Ntor     =   0,  1,  2,  2,  3,  3,  4
   Mnodes   =  [9, 12, 15, 15, 16, 17, 18]
   Nnodes   =  [0;  2;  3;  3;  4;  5;  6]
   
   # continuation parameters
   bdry_ratio = 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0
   pres_ratio = 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0
   zeta_ratio = 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0
   errr_ratio = 1e-3
   pert_order = 1
   
   # solver tolerances
   ftol = 1e-6
   xtol = 1e-6
   gtol = 1e-6
   nfev = 250
   
   # solver methods
   errr_mode = force
   bdry_mode = spectral
   zern_mode = fringe
   node_mode = cheb1
   
   # pressure and rotational transform profiles
   l:   0   cP =   1.80000000E+04   cI =   1.00000000E+00
   l:   2   cP =  -3.60000000E+04   cI =   1.50000000E+00
   l:   4   cP =   1.80000000E+04
   
   # magnetic axis initial guess
   n:   0   aR =   1.00000000E+01   aZ =   0.00000000E+00
   
   # fixed-boundary surface shape
   m:   0   n:   0  bR =   1.00000000E+01   bZ =   0.00000000E+00
   m:   1   n:   0  bR =   1.00000000E+00
   m:   1   n:   1  bR =   3.00000000E-01
   m:  -1   n:  -1  bR =  -3.00000000E-01
   m:  -1   n:   0  bZ =   1.00000000E+00
   m:   1   n:  -1  bZ =  -3.00000000E-01
   m:  -1   n:   1  bZ =  -3.00000000E-01

General Notes
*************

Both ``!`` and ``#`` are recognized to denote comments at the end of a line. 
Whitespace is always ignored, except for newline characters. 
Multiple inputs can be given on the same line of the input file, but a single input cannot span multiple lines. 
None of the inputs are case-sensitive; for example ``Mpol``, ``MPOL``, and ``mPol`` are all the same. 

Global Parameters
*****************

.. code-block:: text

   stell_sym = 1
   NFP       = 19
   Psi_lcfs  = 1.00000000E+00

- ``stell_sym`` (bool): True (1) to assume stellarator symmetry, False (0) otherwise. Default = 0. 
- ``NFP`` (int): Number of toroidal field periods, :math:`N_{FP}`. Default = 1. 
- ``Psi_lcfs`` (float): The toroidal magnetic flux through the last closed flux surface, :math:`\psi_a` (Webers). Default = 1.0. 

Spectral Resolution
*******************

.. code-block:: text

   Mpol     =   6   8  10  10  11  11  12
   delta_lm =   0   4   8  12  16  20  24
   Ntor     =   0,  1,  2,  2,  3,  3,  4
   Mnodes   =  [9, 12, 15, 15, 16, 17, 18]
   Nnodes   =  [0;  2;  3;  3;  4;  5;  6]

- ``Mpol`` (int): Maximum poloidal mode number for the Zernike polynomial basis, :math:`M`. Required. 
- ``delta_lm`` (int): Maximum difference between the radial mode number :math:`l` and the poloidal mode number :math: `m`. Default = ``M`` if ``zern_mode`` is ``ansi`` or ``chevron``, or ``2*M`` if ``zern_mode`` is ``fringe`` or ``house``. For more information see :ref:`theory_zernike_indexing`. 
- ``Ntor`` (int): Maximum toroidal mode number for the Fourier series, :math:`N`. Default = 0. 
- ``Mnodes`` (int): Relative poloidal density of collocation nodes. Default = ``round(1.5*Mpol)``. 
- ``Nnodes`` (int): Relative toroidal density of collocation nodes. Default = ``round(1.5*Ntor)``. 

When ``Mnodes = Mpol`` the number of collocation nodes in each toroidal cross-section is equal to the number of Zernike polynomial in the basis set. 
When ``Nnodes = Ntor`` the number of nodes with unique toroidal angles is equal to the number of terms in the toroidal Fourier series. 
Convergence is typically superior when the number of nodes exceeds the number of spectral coefficients, but this adds compuational cost. 

These arguments can be passed as arrays, where each index of the array denotes the value to use at that iteration. 
In this example there will be 7 iterations, so each array must have a length of 7. 
Note that any type of array notation or deliminator is allowed (only the numbers are extracted). 

Continuation Parameters
***********************

.. code-block:: text

   bdry_ratio = 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0
   pres_ratio = 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0
   zeta_ratio = 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0
   errr_ratio = 1e-3
   pert_order = 1

- ``bdry_ratio`` (float): Multiplier on the 3D boundary modes. Default = 1.0. 
- ``pres_ratio`` (float): Multiplier on the pressure profile. Default = 1.0. 
- ``zeta_ratio`` (float): Multiplier on the toroidal derivatives. Default = 1.0. 
- ``errr_ratio`` (float): Weight on the force balance equations, relative to the boundary condition equations. Default = 1e-2. 
- ``pert_order`` (int): Order of the perturbation approximation: 0 = no perturbation, 1 = linear, 2 = quadratic. Default = 1. 

When all of the ``_ratio`` parameters are set to 1.0, the equilibrium is solved using the exact boundary modes and pressure profile as was input. 
``bdry_ratio = 0`` ignores all of the non-axisymmetric modes, ``pres_ratio = 0`` assumes a vacuum pressure profile, and ``zeta_ratio = 0`` is equivalent to solving a tokamak equilibrium at each toroidal cross-section. 

The fixed-boundary surface shape input is not explicitly enforced. 
If a solution converges to an equilibrium with a different boundary than the one intended, try decreasing ``errr_ratio``. 

These arguments are also passed as arrays for each iteration. 
If only one value is given, as with ``errr_ratio`` and ``pert_order`` in this example, that value will be used for all iterations. 

Solver Tolerances
*****************

.. code-block:: text

   ftol = 1e-6
   xtol = 1e-6
   gtol = 1e-6
   nfev = 250

- ``ftol`` (float): Solver stopping tolerance on relative norm of dF. Default = 1e-6. 
- ``xtol`` (float): Solver stopping tolerance on relative norm of dx. Default = 1e-6. 
- ``gtol`` (float): Solver stopping tolerance on norm of the gradient. Default = 1e-6. 
- ``nfev`` (int): Maximum number of function evaluations. Default = None (0). 

These arguments are also passed as arrays for each iteration. 

Solver Methods
**************

.. code-block:: text

   errr_mode = force
   bdry_mode = spectral
   zern_mode = fringe
   node_mode = cheb1

- ``errr_mode`` (string): Form of equations to use for solving the equilibrium force balance. Options are ``'force'`` (Default) or ``'accel'``. 
- ``bdry_mode`` (string): Form of equations to use for solving the boundary condition. Options are ``'spectral'`` (Default) or ``'real'``. 
- ``zern_mode`` (string): Zernike polynomial index ordering. Options are ``ansi``, ``chevron``, ``house``,  or ``fringe`` (Default). For more information see :ref:`theory_zernike_indexing`. 
- ``node_mode`` (string): Pattern of collocation nodes. Options are ``'cheb1'`` (Default), ``'cheb2'``, or ``'linear'`` (not recommended). 

The ``errr_mode`` option ``'force'`` minimizes the equilibrium force balance errors in units of Newtons, while the ``'accel'`` option uses units of m/radian^2. 
The ``bdry_mode`` option ``'spectral'`` evaluates the error in the boundary condition in Fourier space, while the ``'real'`` option evaluates the error in real space. 

The ``zern_mode`` option ``'ansi'`` uses the OSA/ANSI standard indicies, which has a radial resolution of :math:`M` (the highest radial polynomial term is :math:`\rho^{M}`). 
The ``'fringe'`` option uses the Fringe/University of Arizona indicies, which has a radial resolution of :math:`2M` (the highest radial polynomial term is :math:`\rho^{2M}`). 

All of the node patters use linear spacing in the poloidal and toroidal dimensions. 
The ``'cheb1'`` option places the radial coordinates at the Chebyshev extreme points scaled to the domain [0,1]. 
In this case the collocation nodes are clustered near the magnetic axis and the last closed flux surface. 
The ``'cheb2'`` option places the radial coordinates at the Chebyshev extreme points on the usual domain [-1,1]. 
In this case the collocation nodes are least dense near the magnetic axis and clustered near the last closed flux surface. 
The ``'linear'`` option uses linear spacing for the radial coordinates. 

Pressure & Rotational Transform Profiles
****************************************

.. code-block:: text

   l:   0   cP =   1.80000000E+04   cI =   1.00000000E+00
   l:   2   cP =  -3.60000000E+04   cI =   1.50000000E+00
   l:   4   cP =   1.80000000E+04

- ``l`` (int): Radial polynomial order. 
- ``cP`` (float): Pressure profile coefficient. 
- ``cI`` (float): Rotational transform coefficient. 

The pressure and rotational transform profiles are given as a power series in the flux surface label 
:math:`\rho \equiv \sqrt{\psi / \psi_a}` as follows: 

.. math::

   \begin{aligned}
   p(\rho) &= \sum p_{l} \rho^{l} \\
   \iota(\rho) &= \sum \iota_{l} \rho^{l}.
   \end{aligned}

The coefficients :math:`p_{l}` and :math:`\iota_{l}` are specified by the input variables ``cP`` and ``cI``, respectively. 
The radial exponent :math:`l` is given by ``l``, which must be on the same input line as the coefficients. 
The profiles given in the example are: 

.. math::

   \begin{aligned}
   p(\rho) &= 1.8\times10^4 (1-\rho^2)^2 \\
   \iota(\rho) &= 1 + 1.5 \rho^2.
   \end{aligned}

If no profile inputs are given, it is assumed that they are :math:`p(\rho) = 0` and :math:`\iota(\rho) = 0`. 

Magnetic Axis Initial Guess
***************************

.. code-block:: text

   n:   0   aR =   1.00000000E+01   aZ =   0.00000000E+00

- ``n`` (int): Toroidal mode number. 
- ``aR`` (float): Fourier coefficient of the R coordinate of the magnetic axis. 
- ``aZ`` (float): Fourier coefficient of the Z coordinate of the magnetic axis. 

An initial guess for the magnetic axis can be supplied in the form: 

.. math::

   \begin{aligned}
   R^{a}(\phi) &= \sum_{n=-N}^{N} R^{a}_{n} \mathcal{F}_{n}(\phi) \\
   Z^{a}(\phi) &= \sum_{n=-N}^{N} Z^{a}_{n} \mathcal{F}_{n}(\phi) \\
   \mathcal{F}_{n}(\phi) &= \begin{cases}
   \cos(|n|N_{FP}\phi) &\text{for }n\ge0 \\
   \sin(|n|N_{FP}\phi) &\text{for }n<0. \\
   \end{cases}
   \end{aligned}

The coefficients :math:`R^{a}_{n}` and :math:`Z^{a}_{n}` are specified by the input variables ``aR`` and ``aZ``, respectively. 
The Fourier mode number :math:`n` is given by ``n``, which must be on the same input line as the coefficients. 

If no initial guess is provided for the magnetic axis, then the :math:`m = 0` modes of the fixed-boundary surface shape input are used. 

Fixed-Boundary Surface Shape
****************************

.. code-block:: text

   m:   0   n:   0  bR =   1.00000000E+01  bZ =   0.00000000E+00
   m:   1   n:   0  bR =   1.00000000E+00
   m:   1   n:   1  bR =   3.00000000E-01
   m:  -1   n:  -1  bR =  -3.00000000E-01
   m:  -1   n:   0  bZ =   1.00000000E+00
   m:   1   n:  -1  bZ =  -3.00000000E-01
   m:  -1   n:   1  bZ =  -3.00000000E-01

- ``m`` (int): Poloidal mode number. 
- ``n`` (int): Toroidal mode number. 
- ``bR`` (float): Fourier coefficient of the R coordinate of the last closed flux surface. 
- ``bZ`` (float): Fourier coefficient of the Z coordinate of the last closed flux surface. 

The shape of the fixed-boundary surface is given as a double Fourier series of the form: 

.. math::

   \begin{aligned}
   R^{b}(\theta,\phi) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} R^{b}_{mn} \mathcal{G}^{m}_{n}(\theta,\phi) \\
   Z^{b}(\theta,\phi) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} Z^{b}_{mn} \mathcal{G}^{m}_{n}(\theta,\phi) \\
   \mathcal{G}^{m}_{n}(\theta,\phi) &= \begin{cases}
   \cos(|m|\theta)\cos(|n|N_{FP}\phi) &\text{for }m\ge0, n\ge0 \\
   \cos(|m|\theta)\sin(|n|N_{FP}\phi) &\text{for }m\ge0, n<0 \\
   \sin(|m|\theta)\cos(|n|N_{FP}\phi) &\text{for }m<0, n\ge0 \\
   \sin(|m|\theta)\sin(|n|N_{FP}\phi) &\text{for }m<0, n<0.
   \end{cases}
   \end{aligned}

The coefficients :math:`R^{b}_{mn}` and :math:`Z^{b}_{mn}` are specified by the input variables ``bR`` and ``bZ``, respectively. 
The Fourier mode numbers :math:`m` and :math:`n` are given by ``m`` and ``n``, respectively, which must be on the same input line as the coefficients. 
The fixed-boundary surface shape given in the example is equivalent to (using Ptolemy’s identities):

.. math::

   \begin{aligned}
   R^{b}(\theta,\phi) &= 10 + \cos\theta + 0.3 \cos(\theta+19\phi) \\
   Z^{b}(\theta,\phi) &= \sin\theta - 0.3 \sin(\theta+19\phi).
   \end{aligned}

The fixed-boundary surface shape is a required input. 

VMEC Inputs
***********

A VMEC input file can also be passed in place of a DESC input file. 
DESC will detect if it is a VMEC input format and automatically generate an equivalent DESC input file. 
The generated DESC input file will be stored at the same file path as the VMEC input file, but its name will have ``_desc`` appended to it. 
The resulting input file will not contain any of the options that are specific to DESC, and therefore will depend on many default values. 
This is a convenient first-attempt, but may not converge to the desired result for all equilibria. 
It is recommended that the automatically generated DESC input file be manually edited to improve performance. 
