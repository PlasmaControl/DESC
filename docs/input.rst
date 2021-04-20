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
   sym = 1
   NFP = 19
   Psi = 1.00000000E+00
   
   # spectral resolution
   L_rad  =   0   4   8  12  16  20  24
   M_pol  =   6,  8, 10, 10, 11, 11, 12
   N_tor  =   0;  1;  2;  2;  3;  3;  4
   M_grid =  [9, 12, 15, 15, 16, 17, 18]
   N_grid =  [0;  2;  3;  3;  4;  5;  6]
   
   # continuation parameters
   bdry_ratio = 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0
   pres_ratio = 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0
   pert_order = 2
   
   # solver tolerances
   ftol = 1e-6
   xtol = 1e-6
   gtol = 1e-6
   nfev = 250
   
   # solver methods
   objective = force
   optimizer = scipy-trf
   spectral_indexing = fringe
   node_pattern = jacobi
   
   # pressure and rotational transform profiles
   l:   0   p =  1.80000000E+04   i =  1.00000000E+00
   l:   2   p = -3.60000000E+04   i =  1.50000000E+00
   l:   4   p =  1.80000000E+04
   
   # magnetic axis initial guess
   n:   0  R0 =  1.00000000E+01  Z0 =  0.00000000E+00
   
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

Global Parameters
*****************

.. code-block:: text

   sym = 1
   NFP = 19
   Psi = 1.00000000E+00

- ``sym`` (bool): True (1) to assume stellarator symmetry, False (0) otherwise. Default = 0. 
- ``NFP`` (int): Number of toroidal field periods, :math:`N_{FP}`. Default = 1. 
- ``Psi`` (float): The toroidal magnetic flux through the last closed flux surface, :math:`\psi_a` (Webers). Default = 1.0. 

Spectral Resolution
*******************

.. code-block:: text

   L_rad  =   0   4   8  12  16  20  24
   M_pol  =   6,  8, 10, 10, 11, 11, 12
   N_tor  =   0;  1;  2;  2;  3;  3;  4
   M_grid =  [9, 12, 15, 15, 16, 17, 18]
   N_grid =  [0;  2;  3;  3;  4;  5;  6]

- ``L_rad`` (int): Maximum difference between the radial mode number :math:`l` and the poloidal mode number :math: `m`. Default = ``M`` if ``spectral_indexing`` is ``ansi``, or ``2M`` if ``spectral_indexing`` is ``fringe``. For more information see `Basis functions and collocation nodes`_.
- ``M_pol`` (int): Maximum poloidal mode number for the Zernike polynomial basis, :math:`M`. Required. 
- ``N_tor`` (int): Maximum toroidal mode number for the Fourier series, :math:`N`. Default = 0. 
- ``M_grid`` (int): Relative poloidal density of collocation nodes. Default = ``round(1.5*Mpol)``. 
- ``N_grid`` (int): Relative toroidal density of collocation nodes. Default = ``round(1.5*Ntor)``. 

When ``M_grid = M_pol`` the number of collocation nodes in each toroidal cross-section is equal to the number of Zernike polynomial in the basis set. 
When ``N_grid = N_tor`` the number of nodes with unique toroidal angles is equal to the number of terms in the toroidal Fourier series. 
Convergence is typically superior when the number of nodes exceeds the number of spectral coefficients, but this adds compuational cost. 

These arguments can be passed as arrays, where each index of the array denotes the value to use at that iteration. 
In this example there will be 7 iterations, so each array must have a length of 7. 
Note that any type of array notation or deliminator is allowed (only the numbers are extracted). 

Continuation Parameters
***********************

.. code-block:: text

   bdry_ratio = 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0
   pres_ratio = 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0
   pert_order = 2

- ``bdry_ratio`` (float): Multiplier on the 3D boundary modes. Default = 1.0. 
- ``pres_ratio`` (float): Multiplier on the pressure profile. Default = 1.0. 
- ``pert_order`` (int): Order of the perturbation approximation: 0 = no perturbation, 1 = linear, 2 = quadratic. Default = 1. 

When all of the ``_ratio`` parameters are set to 1.0, the equilibrium is solved using the exact boundary modes and pressure profile as was input. 
``bdry_ratio = 0`` ignores all of the non-axisymmetric modes, and ``pres_ratio = 0`` assumes a vacuum pressure profile. 

These arguments are also passed as arrays for each iteration. 
If only one value is given, as with ``pert_order`` in this example, that value will be used for all iterations. 

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

   objective         = force
   optimizer         = scipy-trf
   spectral_indexing = ansi
   node_pattern      = jacobi

- ``objective`` (string): Form of equations to use for solving the equilibrium. Options are ``'force'`` (Default) or ``'energy'``. 
- ``optimizer`` (string): Type of optimizer to use. For more details and options see :py:class:`desc.optimize.Optimizer`.
- ``spectral_indexing`` (string): Zernike polynomial index ordering. Options are ``ansi`` or ``fringe`` (Default). For more information see `Basis functions and collocation nodes`_.
- ``node_pattern`` (string): Pattern of collocation nodes. Options are ``'jacobi`` (Default), ``cheb1``, ``'cheb2`` or ``'quad``. For more information see `Basis functions and collocation nodes`_.

The ``objective`` option ``'force'`` minimizes the equilibrium force balance errors in units of Newtons, while the ``'energy'`` minimizes the total plasma energy :math:`B^2/2\mu_0 + p`. 

Pressure & Rotational Transform Profiles
****************************************

.. code-block:: text

   l:   0   p =  1.80000000E+04   i =  1.00000000E+00
   l:   2   p = -3.60000000E+04   i =  1.50000000E+00
   l:   4   p =  1.80000000E+04

- ``l`` (int): Radial polynomial order. 
- ``p`` (float): Pressure profile coefficient. :math:`p_{l}` 
- ``i`` (float): Rotational transform coefficient. :math:`\iota_{l}` 

The pressure and rotational transform profiles are given as a power series in the flux surface label 
:math:`\rho \equiv \sqrt{\psi / \psi_a}` as follows: 

.. math::

   \begin{aligned}
   p(\rho) &= \sum p_{l} \rho^{l} \\
   \iota(\rho) &= \sum \iota_{l} \rho^{l}.
   \end{aligned}

The coefficients :math:`p_{l}` and :math:`\iota_{l}` are specified by the input variables ``p`` and ``i``, respectively. 
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

   n:   0  R0 =  1.00000000E+01  Z0 =  0.00000000E+00

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
- ``n`` (int): Toroidal mode number. 
- ``R1`` (float): Fourier coefficient of the R coordinate of the last closed flux surface. :math:`R^{1}_{mn}` 
- ``Z1`` (float): Fourier coefficient of the Z coordinate of the last closed flux surface. :math:`Z^{1}_{mn}` 

The shape of the fixed-boundary surface is given as a double Fourier series of the form: 

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
The Fourier mode numbers :math:`m` and :math:`n` are given by ``m`` and ``n``, respectively, which must be on the same input line as the coefficients. 
The fixed-boundary surface shape given in the example is equivalent to (using Ptolemy’s identities):

.. math::

   \begin{aligned}
   R_{1}(\theta,\phi) &= 10 + \cos\theta + 0.3 \cos(\theta+19\phi) \\
   Z_{1}(\theta,\phi) &= \sin\theta - 0.3 \sin(\theta+19\phi).
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

.. _Basis functions and collocation nodes: notebooks/basis_grid.ipynb
