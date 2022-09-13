
The text output are ASCII files with the naming convention
``output.FILE_NAME``. All of the necessary variables to fully define an
equilibrium solution are output in the following order: grid parameters,
fixed-boundary shape, pressure and rotational transform profiles, flux
surface shapes, and the boundary function :math:`\lambda`. An example
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
   flux surface, :math:`\psi_a`, in Webers

Fixed-Boundary Shape
--------------------

The target shape of the plasma boundary is output for reference in the
section of the output file with the heading ``Nbdry``. This gives the number
of boundary terms, followed by the coefficients. This is the fixed-boundary
input that was used to compute the equilibrium, but the
last closed flux surface generally does not match this desired shape
exactly. The shape of the boundary surface is given as a double Fourier
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

Flux Surface Shapes
-------------------

The shapes of the flux surfaces are the solution to the equilibrium
defined by the fixed-boundary and profile inputs. They are given by a
Fourier-Zernike basis set with “fringe” indexing of the form:


.. math::

   \begin{aligned}
   R(\rho,\vartheta,\zeta) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} \sum_{l\in L} R_{lmn} \mathcal{Z}^{m}_{l}(\rho,\vartheta) \mathcal{F}^{n}(\zeta) \\
   Z(\rho,\vartheta,\zeta) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} \sum_{l\in L} Z_{lmn} \mathcal{Z}^{m}_{l}(\rho,\vartheta) \mathcal{F}^{n}(\zeta)\end{aligned}

where :math:`L = |m|, |m|+2, |m|+4, \ldots, 2 M`.
:math:`\mathcal{F}^{n}(\zeta)` is the toroidal Fourier series defined as

.. math::

   \mathcal{F}^{n}(\zeta) = \begin{cases}
   \cos(|n|N_{FP}\zeta) &\text{for }n\ge0 \\
   \sin(|n|N_{FP}\zeta) &\text{for }n<0. \\
   \end{cases}


:math:`\mathcal{Z}^{m}_{l}(\rho,\vartheta)` are the Zernike polynomials
defined on the unit disc :math:`0\leq\rho\leq1`,
:math:`\vartheta\in[0,2\pi)` as

.. math::

   \mathcal{Z}^{m}_{l}(\rho,\vartheta) = \begin{cases}
   \mathcal{R}^{|m|}_{l}(\rho) \cos(|m|\vartheta) &\text{for }m\ge0 \\
   \mathcal{R}^{|m|}_{l}(\rho) \sin(|m|\vartheta) &\text{for }m<0 \\
   \end{cases}

with the radial function

.. math:: \mathcal{R}^{|m|}_{l}(\rho) = \sum^{(l-|m|)/2}_{s=0} \frac{(-1)^s(l-s)!}{s![\frac{1}{2}(l+|m|)-s]![\frac{1}{2}(l-|m|)-s]!} \rho^{l-2s}.

The Fourier-Zernike coefficients :math:`R_{mn}` and :math:`Z_{mn}` are
given by the variables ``cR`` and ``cZ``, respectively, in the section
of the output file with the heading ``NRZ`` (which gives the total number
of values). The indices :math:`l`, :math:`m`, and :math:`n` that identify
each coefficient are given by the variables ``l``, ``m``, and ``n`` on
the same line of the output file as ``cR`` and ``cZ``.
When stellarator symmetry is enforced,
only the :math:`R_{mn}` with :math:`mn > 0` and the :math:`Z_{mn}` with
:math:`mn < 0` are nonzero. Coefficients with :math:`mn = 0` are nonzero
for :math:`R_{mn}` if one of the mode numbers is positive, and nonzero
for :math:`Z_{mn}` if one of the mode numbers is negative. Lines 45-46
of the example output file give the terms

.. math::

   \begin{aligned}
   R_{3,1,1} \mathcal{Z}^{1}_{3}(\rho,\vartheta) \mathcal{F}^{1}(\zeta) &= 5.26674681 \times 10^{-2} (3\rho^3-2\rho) \cos(\vartheta) \cos(3\zeta) \\
   Z_{2,2,-1} \mathcal{Z}^{2}_{2}(\rho,\vartheta) \mathcal{F}^{-1}(\zeta) &= 5.01543691 \times 10^{-2} \rho^2 \cos(2\vartheta) \sin(3\zeta).\end{aligned}

The magnetic field is computed in the straight field-line coordinate
system :math:`(\rho,\vartheta,\zeta)` by

.. math:: \mathbf{B} = B^\vartheta {\mathbf e}_{\vartheta}+ B^\zeta {\mathbf e}_{\zeta}= \frac{2\psi_a \rho}{2\pi \sqrt{g}} \left( \iota {\mathbf e}_{\vartheta}+ {\mathbf e}_{\zeta}\right).

The covariant basis vectors are defined as

.. math:: {\mathbf e}_{\rho}= \begin{bmatrix} \partial_\rho R \\ 0 \\ \partial_\rho Z \end{bmatrix} \hspace{5mm} {\mathbf e}_{\vartheta}= \begin{bmatrix} \partial_\vartheta R \\ 0 \\ \partial_\vartheta Z \end{bmatrix} \hspace{5mm} {\mathbf e}_{\zeta}= \begin{bmatrix} \partial_\zeta R \\ R \\ \partial_\zeta Z \end{bmatrix}

and the Jacobian of the coordinate system is
:math:`\sqrt{g} = {\mathbf e}_{\rho}\cdot{\mathbf e}_{\vartheta}\times{\mathbf e}_{\zeta}`.
The partial derivatives of :math:`R(\rho,\vartheta,\zeta)` and
:math:`Z(\rho,\vartheta,\zeta)` are known analytically from
the basis functions. The components of the magnetic field
in the toroidal coordinate system :math:`(R,\phi,Z)` can be easily
computed as :math:`B_i = \mathbf{B} \cdot \mathbf{e}_i` with
:math:`{\mathbf e}_{R}= [1, 0, 0]^T`,
:math:`{\mathbf e}_{\phi}= [0, 1, 0]^T`, and
:math:`{\mathbf e}_{Z}= [0, 0, 1]^T`.

Boundary Function :math:`\lambda`
---------------------------------

The straight field-line angle :math:`\zeta` is equivalent to the
toroidal angle by definition: :math:`\zeta = \phi`. The function
:math:`\lambda(\theta,\phi)` relates the straight field-line angle
:math:`\vartheta` to the poloidal angle used to define the boundary
surface :math:`\theta` through the equation
:math:`\vartheta = \theta + \lambda(\theta,\phi)`. It is used internally
to enforce the boundary condition at the last closed flux surface, and
is output for reference. The function is given as a doubles Fourier
series of the form:

.. math::

   \begin{aligned}
   \lambda(\theta,\phi) &= \sum_{n=-N}^{N} \sum_{m=-M}^{M} \lambda_{mn} \mathcal{G}^{m}_{n}(\theta,\phi)\end{aligned}

where :math:`\mathcal{G}^{m}_{n}(\theta,\phi)` was defined above for the
boundary shape. The Fourier coefficients :math:`\lambda_{mn}` are
given by the variable ``cL`` in the section of the output file with the
heading ``NL`` (which gives the number of :math:`\lambda` coefficients).
Their output format follows the same convention as
the boundary coefficients ``bR`` and ``bZ``. When stellarator symmetry
is enforced, only the coefficients with :math:`mn < 0` are nonzero.
Coefficients with :math:`mn = 0` are nonzero if one of the mode numbers
is negative.


Example Output File
-------------------

::

   NFP =   3
   Psi =   1.00000000E+00
   Nbdry =  7
   m:   0 n:   0 bR =   1.00000000E+01 bZ =   0.00000000E+00
   m:   1 n:   0 bR =  -1.00000000E+00 bZ =   0.00000000E+00
   m:  -1 n:   0 bR =   0.00000000E+00 bZ =   1.00000000E+00
   m:   1 n:   1 bR =  -3.00000000E-01 bZ =   0.00000000E+00
   m:  -1 n:  -1 bR =  -3.00000000E-01 bZ =   0.00000000E+00
   m:  -1 n:   1 bR =   0.00000000E+00 bZ =  -3.00000000E-01
   m:   1 n:  -1 bR =   0.00000000E+00 bZ =   3.00000000E-01
   Nprof =   5
   l:   0 cP =   3.40000000E+03 cI =   5.00000000E-01
   l:   1 cP =   0.00000000E+00 cI =   0.00000000E+00
   l:   2 cP =  -6.80000000E+03 cI =   1.50000000E+00
   l:   3 cP =   0.00000000E+00 cI =   0.00000000E+00
   l:   4 cP =   3.40000000E+03 cI =   0.00000000E+00
   NRZ =   27
   l:   0 m:   0 n:  -1 cR =   0.00000000E+00 cZ =  -2.90511418E-03
   l:   0 m:   0 n:   0 cR =   9.98274712E+00 cZ =   0.00000000E+00
   l:   0 m:   0 n:   1 cR =  -2.90180674E-03 cZ =   0.00000000E+00
   l:   1 m:  -1 n:  -1 cR =   2.28896490E-01 cZ =   0.00000000E+00
   l:   1 m:  -1 n:   0 cR =   0.00000000E+00 cZ =   9.48092222E-01
   l:   1 m:  -1 n:   1 cR =   0.00000000E+00 cZ =  -2.27403979E-01
   l:   2 m:   0 n:  -1 cR =   0.00000000E+00 cZ =  -2.41707137E-02
   l:   2 m:   0 n:   0 cR =  -1.36531448E-01 cZ =   0.00000000E+00
   l:   2 m:   0 n:   1 cR =  -2.41387024E-02 cZ =   0.00000000E+00
   l:   1 m:   1 n:  -1 cR =   0.00000000E+00 cZ =   2.24346193E-01
   l:   1 m:   1 n:   0 cR =   9.25944834E-01 cZ =   0.00000000E+00
   l:   1 m:   1 n:   1 cR =   2.25843613E-01 cZ =   0.00000000E+00
   l:   2 m:  -2 n:  -1 cR =   3.34519544E-02 cZ =   0.00000000E+00
   l:   2 m:  -2 n:   0 cR =   0.00000000E+00 cZ =   1.58172393E-01
   l:   2 m:  -2 n:   1 cR =   0.00000000E+00 cZ =  -5.03483447E-02
   l:   3 m:  -1 n:  -1 cR =   4.81316537E-02 cZ =   0.00000000E+00
   l:   3 m:  -1 n:   0 cR =   0.00000000E+00 cZ =   3.38024112E-02
   l:   3 m:  -1 n:   1 cR =   0.00000000E+00 cZ =  -4.74860303E-02
   l:   4 m:   0 n:  -1 cR =   0.00000000E+00 cZ =   2.08609498E-02
   l:   4 m:   0 n:   0 cR =   1.33345992E-01 cZ =   0.00000000E+00
   l:   4 m:   0 n:   1 cR =   2.07783052E-02 cZ =   0.00000000E+00
   l:   3 m:   1 n:  -1 cR =   0.00000000E+00 cZ =   5.20291455E-02
   l:   3 m:   1 n:   0 cR =   7.29416666E-02 cZ =   0.00000000E+00
   l:   3 m:   1 n:   1 cR =   5.26674681E-02 cZ =   0.00000000E+00
   l:   2 m:   2 n:  -1 cR =   0.00000000E+00 cZ =   5.01543691E-02
   l:   2 m:   2 n:   0 cR =   1.56388795E-01 cZ =   0.00000000E+00
   l:   2 m:   2 n:   1 cR =   3.32590868E-02 cZ =   0.00000000E+00
   NL =  12
   m:  -2 n:  -1 cL =  -0.00000000E+00
   m:  -2 n:   0 cL =   9.55435813E-03
   m:  -2 n:   1 cL =   2.53333116E-02
   m:  -1 n:  -1 cL =  -0.00000000E+00
   m:  -1 n:   0 cL =   9.91996517E-02
   m:  -1 n:   1 cL =  -1.17417875E-02
   m:   0 n:  -1 cL =   1.75103748E-04
   m:   0 n:   0 cL =  -0.00000000E+00
   m:   0 n:   1 cL =  -0.00000000E+00
   m:   1 n:  -1 cL =   1.16506641E-02
   m:   1 n:   0 cL =  -0.00000000E+00
   m:   1 n:   1 cL =  -0.00000000E+00
