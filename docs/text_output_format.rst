
Limited support for ASCII output files exists with the function :func:`desc.io.write_ascii`.
All of the necessary variables to fully define an
equilibrium solution are output in the following order: grid parameters,
fixed-boundary shape, pressure and rotational transform profiles, flux
surface shapes, and the poloidal stream function :math:`\lambda`. An example
output file is included for reference at the end of this document. All
integers are printed with a total width of 3 characters, and all
floating point numbers are printed in exponential notation with a total
width of 16 characters including 8 digits after the decimal points.

Grid Parameters
---------------

The first two lines of the output file specify some global parameters:
next four lines contain the following information in order:

#. ``NFP`` (integer): number of field periods

#. ``Psi`` (float): total toroidal magnetic flux through the last closed
   flux surface, :math:`\Psi`, in Webers

Fixed-Boundary Shape
--------------------

The target shape of the plasma boundary is output for reference in the
section of the output file with the heading ``Nbdry``. This gives the number
of boundary terms, followed by the coefficients. This is the fixed-boundary surface
input that was used to compute the equilibrium, and is the last closed flux surface of the equilibrium.
The shape of the boundary surface is given as a double Fourier
series of the form:

.. math::

   \begin{aligned}
   R^b(\theta,\phi) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} R^{b}_{mn} \mathcal{G}^{m}_{n}(\theta,\phi) \\
   Z^b(\theta,\phi) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} Z^{b}_{mn} \mathcal{G}^{m}_{n}(\theta,\phi) \\
   \label{eq:G}
   \mathcal{G}^{m}_{n}(\theta,\phi) &= \begin{cases}
   \cos(|m|\theta)\cos(|n|N_{FP}\phi) &\text{for }m\ge0, n\ge0 \\
   \cos(|m|\theta)\sin(|n|N_{FP}\phi) &\text{for }m\ge0, n<0 \\
   \sin(|m|\theta)\cos(|n|N_{FP}\phi) &\text{for }m<0, n\ge0 \\
   \sin(|m|\theta)\sin(|n|N_{FP}\phi) &\text{for }m<0, n<0.
   \end{cases}\end{aligned}

The Fourier coefficients :math:`R^{b}_{mn}` and :math:`Z^{b}_{mn}` are
given by the output variables ``bR`` and ``bZ``, respectively. The
poloidal and toroidal mode numbers :math:`m` and :math:`n` that identify
each coefficient are given by the variables ``m`` and ``n`` on the same
line of the output file as ``bR`` and ``bZ``. When stellarator symmetry
is enforced, only the :math:`R^{b}_{mn}` with :math:`mn > 0` and the
:math:`Z^{b}_{mn}` with :math:`mn < 0` are nonzero. Coefficients with
:math:`mn = 0` are nonzero for :math:`R^{b}_{mn}` if one of the mode
numbers is positive, and nonzero for :math:`Z^{b}_{mn}` if one of the
mode numbers is negative. The boundary surface given in the example is
equivalent to (using Ptolemy’s identities):

.. math::

   \begin{aligned}
   R^b &= 10 - \cos\theta - 0.3 \cos(\theta-3\phi) \\
   Z^b &= \sin\theta - 0.3 \sin(\theta-3\phi).\end{aligned}

Pressure & Rotational Transform Profiles
----------------------------------------

The pressure and rotational transform profiles that were used to compute
the equilibrium are also output for reference in the section of the
output file with the heading ``Nprof``, which also gives the number of profile
coefficients. These are given as a power series in the flux surface label
:math:`\rho \equiv \sqrt{\psi / \psi_a}` as follows:

.. math::

   \begin{aligned}
   p(\rho) &= \sum_{l=0}^{2M} p_{l} \rho^{l} \\
   \iota(\rho) &= \sum_{l=0}^{2M} \iota_{l} \rho^{l}.\end{aligned}

The coefficients :math:`p_{l}` and :math:`\iota_{l}` are given by the
output variables ``cP`` and ``cI``, respectively. The radial order
:math:`l` that identifies each coefficient is given by the variable
``l`` on the same line of the output file as ``cP`` and ``cI``. The
profiles given in the example are:

.. math::

   \begin{aligned}
   p &= 3.4\times10^3 (1-\rho^2)^2 \\
   \iota &= 0.5 + 1.5 \rho^2.\end{aligned}

If the equilibrium was solved with a fixed-current profile, then the iota
computed from the equilibrium will be fit with a power series, and the coefficients of
that power series will be saved. Likewise, if the equilibrium does not have a pressure profile
but rather kinetic profiles, the pressure will be computed, fit with a power series, and saved.

Flux Surface Shapes
-------------------

The shapes of the flux surfaces are the solution to the equilibrium
defined by the fixed-boundary and profile inputs. They are given by a
Fourier-Zernike basis set with “fringe” indexing of the form:


.. math::

   \begin{aligned}
   R(\rho,\theta,\zeta) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} \sum_{l\in L} R_{lmn} \mathcal{Z}^{m}_{l}(\rho,\theta) \mathcal{F}^{n}(\zeta) \\
   Z(\rho,\theta,\zeta) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} \sum_{l\in L} Z_{lmn} \mathcal{Z}^{m}_{l}(\rho,\theta) \mathcal{F}^{n}(\zeta)\end{aligned}

where :math:`L = |m|, |m|+2, |m|+4, \ldots, 2 M`.
:math:`\mathcal{F}^{n}(\zeta)` is the toroidal Fourier series defined as

.. math::

   \mathcal{F}^{n}(\zeta) = \begin{cases}
   \cos(|n|N_{FP}\zeta) &\text{for }n\ge0 \\
   \sin(|n|N_{FP}\zeta) &\text{for }n<0. \\
   \end{cases}


:math:`\mathcal{Z}^{m}_{l}(\rho,\theta)` are the Zernike polynomials
defined on the unit disc :math:`0\leq\rho\leq1`,
:math:`\theta\in[0,2\pi)` as

.. math::

   \mathcal{Z}^{m}_{l}(\rho,\theta) = \begin{cases}
   \mathcal{R}^{|m|}_{l}(\rho) \cos(|m|\theta) &\text{for }m\ge0 \\
   \mathcal{R}^{|m|}_{l}(\rho) \sin(|m|\theta) &\text{for }m<0 \\
   \end{cases}

with the radial function

.. math:: \mathcal{R}^{|m|}_{l}(\rho) = \sum^{(l-|m|)/2}_{s=0} \frac{(-1)^s(l-s)!}{s![\frac{1}{2}(l+|m|)-s]![\frac{1}{2}(l-|m|)-s]!} \rho^{l-2s}.

The Fourier-Zernike coefficients :math:`R_{lmn}`,  :math:`Z_{lmn}` and :math:`\lambda_lmn` are
given by the variables ``cR``, ``cZ``, and ``cL`` , respectively, in the section
of the output file with the heading ``NRZ`` (which gives the total number
of values). The indices :math:`l`, :math:`m`, and :math:`n` that identify
each coefficient are given by the variables ``l``, ``m``, and ``n`` on
the same line of the output file as ``cR``, ``cZ`` and ``cL``.
When stellarator symmetry is enforced,
only the :math:`R_{lmn}` with :math:`m,n > 0` and the :math:`Z_{lmn}` and :math:`\lambda_{lmn}` with
:math:`m,n < 0` are nonzero. Coefficients with :math:`m,n = 0` are nonzero
for :math:`R_{lmn}` if one of the mode numbers is positive, and nonzero
for :math:`Z_{lmn}` and :math:`\lambda_{lmn}` if one of the mode numbers is negative. Lines 45-46
of the example output file give the terms

.. math::

   \begin{aligned}
   R_{3,1,1} \mathcal{Z}^{1}_{3}(\rho,\theta) \mathcal{F}^{1}(\zeta) &= 5.26674681 \times 10^{-2} (3\rho^3-2\rho) \cos(\theta) \cos(3\zeta) \\
   Z_{2,2,-1} \mathcal{Z}^{2}_{2}(\rho,\theta) \mathcal{F}^{-1}(\zeta) &= 5.01543691 \times 10^{-2} \rho^2 \cos(2\theta) \sin(3\zeta).\end{aligned}


Example Output File
-------------------

::

   NFP =  19
   Psi =   1.00000000E+00
   Nbdry =  24
   m:  -2 n:  -2 bR =   1.75911802E-17 bZ =   0.00000000E+00
   m:  -1 n:  -2 bR =   8.80914265E-19 bZ =   0.00000000E+00
   m:  -2 n:  -1 bR =   5.42101086E-19 bZ =   0.00000000E+00
   m:  -1 n:  -1 bR =   3.00000000E-01 bZ =   0.00000000E+00
   m:   0 n:   0 bR =   1.00000000E+01 bZ =   0.00000000E+00
   m:   1 n:   0 bR =  -1.00000000E+00 bZ =   0.00000000E+00
   m:   2 n:   0 bR =  -1.02999206E-18 bZ =   0.00000000E+00
   m:   0 n:   1 bR =   1.68051337E-18 bZ =   0.00000000E+00
   m:   1 n:   1 bR =  -3.00000000E-01 bZ =   0.00000000E+00
   m:   2 n:   1 bR =   2.13791116E-18 bZ =   0.00000000E+00
   m:   0 n:   2 bR =  -3.52365706E-19 bZ =   0.00000000E+00
   m:   1 n:   2 bR =  -1.08420217E-19 bZ =   0.00000000E+00
   m:   2 n:   2 bR =   1.35525272E-19 bZ =   0.00000000E+00
   m:   0 n:  -2 bR =   0.00000000E+00 bZ =   5.42101086E-20
   m:   1 n:  -2 bR =   0.00000000E+00 bZ =  -8.13151629E-20
   m:   2 n:  -2 bR =   0.00000000E+00 bZ =   2.50721752E-19
   m:   0 n:  -1 bR =   0.00000000E+00 bZ =   9.35124374E-18
   m:   1 n:  -1 bR =   0.00000000E+00 bZ =  -3.00000000E-01
   m:   2 n:  -1 bR =   0.00000000E+00 bZ =  -6.09863722E-18
   m:  -1 n:   0 bR =   0.00000000E+00 bZ =   1.00000000E+00
   m:  -2 n:   1 bR =   0.00000000E+00 bZ =   1.83636743E-18
   m:  -1 n:   1 bR =   0.00000000E+00 bZ =  -3.00000000E-01
   m:  -2 n:   2 bR =   0.00000000E+00 bZ =   1.89735380E-19
   m:  -1 n:   2 bR =   0.00000000E+00 bZ =   2.84603070E-19
   Nprof =   3
   l:   0 cP =   1.80000000E+04 cI =   1.00000000E+00
   l:   2 cP =  -3.60000000E+04 cI =   1.50000000E+00
   l:   4 cP =   1.80000000E+04 cI =   0.00000000E+00
   NRZ =    40
   l:   1 m:  -1 n:  -2 cR =  -2.05652967E-03 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   2 m:  -2 n:  -2 cR =  -6.91124500E-04 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   3 m:  -1 n:  -2 cR =   8.88430363E-04 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   1 m:  -1 n:  -1 cR =   2.29131774E-01 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   2 m:  -2 n:  -1 cR =  -5.22612182E-03 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   3 m:  -1 n:  -1 cR =   3.73498631E-02 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   0 m:   0 n:   0 cR =   1.01304079E+01 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   1 m:   1 n:   0 cR =  -9.71393329E-01 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   2 m:   0 n:   0 cR =  -1.78433121E-01 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   2 m:   2 n:   0 cR =   1.61065486E-02 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   3 m:   1 n:   0 cR =  -3.13199707E-02 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   0 m:   0 n:   1 cR =   3.98162495E-02 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   1 m:   1 n:   1 cR =  -2.51860081E-01 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   2 m:   0 n:   1 cR =  -4.79783330E-02 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   2 m:   2 n:   1 cR =   9.54103619E-03 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   3 m:   1 n:   1 cR =  -2.68561658E-02 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   0 m:   0 n:   2 cR =   8.29727328E-05 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   1 m:   1 n:   2 cR =   9.35653958E-04 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   2 m:   0 n:   2 cR =  -7.00132972E-04 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   2 m:   2 n:   2 cR =   1.08982466E-03 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   3 m:   1 n:   2 cR =  -3.83056775E-04 cZ =   0.00000000E+00 cL =   0.00000000E+00
   l:   0 m:   0 n:  -2 cR =   0.00000000E+00 cZ =   7.85251369E-05 cL =  -2.99837951E-04
   l:   1 m:   1 n:  -2 cR =   0.00000000E+00 cZ =   8.86922440E-04 cL =  -4.43171034E-04
   l:   2 m:   0 n:  -2 cR =   0.00000000E+00 cZ =  -7.47840516E-04 cL =  -1.92494857E-04
   l:   2 m:   2 n:  -2 cR =   0.00000000E+00 cZ =   1.14169192E-03 cL =  -5.62839907E-05
   l:   3 m:   1 n:  -2 cR =   0.00000000E+00 cZ =  -2.20111383E-04 cL =   1.04591491E-04
   l:   0 m:   0 n:  -1 cR =   0.00000000E+00 cZ =   4.06168259E-02 cL =  -1.94375807E-02
   l:   1 m:   1 n:  -1 cR =   0.00000000E+00 cZ =  -2.64784931E-01 cL =   3.17712917E-03
   l:   2 m:   0 n:  -1 cR =   0.00000000E+00 cZ =  -4.54725804E-02 cL =   1.15712124E-02
   l:   2 m:   2 n:  -1 cR =   0.00000000E+00 cZ =   6.02610577E-03 cL =  -4.35783380E-02
   l:   3 m:   1 n:  -1 cR =   0.00000000E+00 cZ =  -1.62018350E-02 cL =  -2.14132868E-02
   l:   1 m:  -1 n:   0 cR =   0.00000000E+00 cZ =   9.69654126E-01 cL =   6.33862049E-01
   l:   2 m:  -2 n:   0 cR =   0.00000000E+00 cZ =  -1.52205137E-02 cL =   6.72394132E-03
   l:   3 m:  -1 n:   0 cR =   0.00000000E+00 cZ =  -9.78453786E-03 cL =  -4.25692127E-01
   l:   1 m:  -1 n:   1 cR =   0.00000000E+00 cZ =  -2.46867402E-01 cL =   1.19756872E-02
   l:   2 m:  -2 n:   1 cR =   0.00000000E+00 cZ =   4.38597594E-03 cL =  -5.03481139E-02
   l:   3 m:  -1 n:   1 cR =   0.00000000E+00 cZ =  -2.63354652E-02 cL =  -4.56728374E-02
   l:   1 m:  -1 n:   2 cR =   0.00000000E+00 cZ =   1.81476307E-03 cL =  -3.97405295E-04
   l:   2 m:  -2 n:   2 cR =   0.00000000E+00 cZ =   7.27589399E-04 cL =  -2.60842517E-04
   l:   3 m:  -1 n:   2 cR =   0.00000000E+00 cZ =  -6.41618803E-04 cL =  -6.99577466E-06
