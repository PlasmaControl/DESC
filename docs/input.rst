======
Inputs
======

.. role:: bash(code)
   :language: bash

The following is an example DESC input file, which containts all of the available input arguments. 
More input examples are included in the repository. 
DESC can also accept VMEC input files, which are converted to DESC inputs as explained below. 

.. code-block:: text
   :linenos:

   ! this is a comment
   # this is also a comment
   
   # global parameters
   stell_sym	= 1
   NFP			= 19
   Psi_lcfs	= 1.00000000E+00
   
   # spectral resolution
   Mpol		=   6   8  10  10  11  11  12
   Ntor		=   0,  1,  2,  2,  3,  3,  4
   Mnodes	=  [9, 12, 15, 15, 16, 17, 18]
   Nnodes	=  [0;  2;  3;  3;  4;  5;  6]
   
   # continuation parameters
   bdry_ratio	= 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0
   pres_ratio	= 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0
   zeta_ratio	= 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0
   errr_ratio	= 1e-3
   pert_order	= 1
   
   # solver tolerances
   ftol	= 1e-6
   xtol	= 1e-6
   gtol	= 1e-6
   nfev	= 250
   
   # solver methods
   errr_mode	= force
   bdry_mode	= spectral
   node_mode	= cheb2
   
   # pressure and rotational transform profiles
   l:   0	cP =   1.80000000E+04	cI =   1.00000000E+00
   l:   2	cP =  -3.60000000E+04	cI =   1.50000000E+00
   l:   4	cP =   1.80000000E+04
   
   # magnetic axis initial guess
   n:   0	aR =   1.00000000E+01	aZ =   0.00000000E+00
   
   # fixed-boundary surface shape
   m:   0	n:   0	bR =   1.00000000E+01	bZ =   0.00000000E+00
   m:   1	n:   0	bR =   1.00000000E+00
   m:   1	n:   1	bR =   3.00000000E-01
   m:  -1	n:  -1	bR =  -3.00000000E-01
   m:  -1	n:   0	bZ =   1.00000000E+00
   m:   1	n:  -1	bZ =  -3.00000000E-01
   m:  -1	n:   1	bZ =  -3.00000000E-01

Note that both `#` and `!` are recognized to denote comments at the end of a line. 
Whitespace is always ignored, except for newline characters. 
Multiple inputs can be given on the same line of the input file, but a single input cannot span multiple lines. 

Global Parameters
*****************

.. code-block:: text
   :linenos:

   stell_sym	= 1				! stellarator symmetry
   NFP			= 19			! number of field periods
   Psi_lcfs	= 1.00000000E+00	! toroidal flux at the last closed flux surface

- `stell_sym` (bool): True (1) to assume stellarator symmetry, False (0) otherwise. Default = 0. 
- `NFP` (int): Number of toroidal field periods. Default = 1. 
- `Psi_lcfs` (float): The toroidal magnetic flux through the last closed flux surface, in Webers. Default = 1.0. 

Spectral Resolution
*******************

.. code-block:: text
   :linenos:

   Mpol		=   6   8  10  10  11  11  12
   Ntor		=   0,  1,  2,  2,  3,  3,  4
   Mnodes	=  [9, 12, 15, 15, 16, 17, 18]
   Nnodes	=  [0;  2;  3;  3;  4;  5;  6]

Continuation Parameters
***********************

.. code-block:: text
   :linenos:

   bdry_ratio	= 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0
   pres_ratio	= 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0
   zeta_ratio	= 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0
   errr_ratio	= 1e-3
   pert_order	= 1

Solver Tolerances
*****************

.. code-block:: text
   :linenos:

   ftol	= 1e-6
   xtol	= 1e-6
   gtol	= 1e-6
   nfev	= 250

Solver Methods
**************

.. code-block:: text
   :linenos:

   errr_mode	= force
   bdry_mode	= spectral
   node_mode	= cheb2

Pressure & Rotational Transform Profiles
****************************************

.. code-block:: text
   :linenos:

   l:   0	cP =   1.80000000E+04	cI =   1.00000000E+00
   l:   2	cP =  -3.60000000E+04	cI =   1.50000000E+00
   l:   4	cP =   1.80000000E+04

Magnetic Axis Initial Guess
***************************

.. code-block:: text
   :linenos:

   n:   0	aR =   1.00000000E+01	aZ =   0.00000000E+00


Fixed-Boundary Surface Shape
****************************

.. code-block:: text
   :linenos:

   m:   0	n:   0	bR =   1.00000000E+01	bZ =   0.00000000E+00
   m:   1	n:   0	bR =   1.00000000E+00
   m:   1	n:   1	bR =   3.00000000E-01
   m:  -1	n:  -1	bR =  -3.00000000E-01
   m:  -1	n:   0	bZ =   1.00000000E+00
   m:   1	n:  -1	bZ =  -3.00000000E-01
   m:  -1	n:   1	bZ =  -3.00000000E-01

VMEC Inputs
***********

A VMEC input file can also be passed in place of a DESC input file. 
DESC will detect if it is a VMEC input format and automatically generate an equivalent DESC input file. 
The generated DESC input file will be stored at the same file path as the VMEC input file, but its name will have :bash:`_desc` appended to it. 
The resulting input file will not contain any of the options that are specific to DESC, and therefore will depend on many default values. 
This is a convenient first-attempt, but may not converge to the desired result for all equilibria. 
It is recommended that the automatically generated DESC input file be manually edited to improve performance. 
