=======
General
=======

The theoretical approach and numerical methods used by DESC are explained in this publication_ [1]_. 
The code is constantly evolving and may differ from the implementation presented in the original paper. 
This documentation aims to reflect the latest version of the code, and major discrepancies from the publication will be noted when relevant. 

.. [1] D.W. Dudt and E. Kolemen (2020). DESC: A Stellarator Equilibrium Solver. *Physics of Plasmas*. 
.. _publication: https://github.com/PlasmaControl/DESC/blob/master/docs/Dudt_Kolemen_PoP_2020.pdf

See also our recent pre-prints [2-4]:

.. [2] D. Panici, R. Conlin, D.W. Dudt and E. Kolemen. “The DESC Stellarator Code Suite Part I: Quick and accurate equilibria computations.” pre-print.
.. _publication: https://arxiv.org/abs/2203.17173
.. [3] R. Conlin, D.W. Dudt, D. Panici and E. Kolemen. “The DESC Stellarator Code Suite Part II: Perturbation and continuation methods.” pre-print.
.. _publication: https://arxiv.org/abs/2203.15927
.. [4] D.W. Dudt, R. Conlin, D. Panici and E. Kolemen. “The DESC Stellarator Code Suite Part III: Quasi-symmetry optimization.” pre-print.
.. _publication: https://arxiv.org/abs/2204.00078

Flux coordinates
****************

DESC solves the "inverse" equilibrium problem. 
The computational domain is the curvilinear coordinate system :math:`(\rho, \theta, \zeta)`, where :math:`\zeta` is chosen to be the toroidal angle of the cylindrical coordinate system :math:`(R, \phi, Z)`. 
These curvilinear coordinates are related to the straight field-line coordinates :math:`(\rho, \vartheta, \zeta)` through the stream function :math:`\lambda(\rho,\theta,\zeta)`. 
[Note: the original publication used :math:`\zeta=-\phi` and used :math:`\vartheta` in the computational domain instead of introducing :math:`\lambda` on all flux surfaces.] 
This particular choice of flux coordinates is also used by the PEST code, and should not be confused with other choices such as Boozer or Hamada coordinates. 
The flux surface label :math:`\rho` is chosen to be the square root of the normalized toroidal flux, which is proportional to the minor radius. 
This is different from the default radial coordinate in VMEC of the normalized toroidal flux. 

.. image:: _static/images/coordinates.png
  :width: 800
  :alt: Alternative text

The covariant basis vectors of the curvilinear coordinate system are 

.. math::
  \mathbf{e}_\rho = [\partial_\rho R \\ 0 \\ \partial_\rho Z]^T \\ \\ \mathbf{e}_\theta = [\partial_\theta R \\ 0 \\ \partial_\theta Z]^T \\ \\ \mathbf{e}_\zeta = [\partial_\zeta R \\ R \\ \partial_\zeta Z]^T

and the Jacobian of the curvilinear coordinate system is :math:`\sqrt{g} = e_\rho \cdot e_\theta \times e_\zeta`. 

DESC solves for the map between the cylindrical and flux coordinate systems through the following variables that represent the shapes of the flux surfaces: 

.. math::
  R(\rho, \theta, \zeta) \\ \\ Z(\rho, \theta, \zeta) \\ \\ \lambda(\rho, \theta, \zeta)

It assumes the flux functions for the pressure :math:`p(\rho)` and rotational transform :math:`\iota(\rho)` profiles are given, in addition to the total toroidal flux through the plasma volume :math:`\psi_a`. 
The shape of the last closed flux surface :math:`R^b(\theta,\phi)`, :math:`Z^b(\theta,\phi)` is also required to specify the fixed-boundary. 

Magnetic Field & Current Density
********************************

By assuming nested flux surfaces, :math:`\mathbf{B} \cdot \nabla \rho = 0`, and invoking Gauss's Law, :math:`\nabla \cdot \mathbf{B} = 0`, the magnetic field is written in flux coordinates as 

.. math::
  \mathbf{B} = B^\theta \mathbf{e}_\theta + B^\zeta \mathbf{e}_\zeta = \frac{\partial_\rho \psi}{2 \pi \sqrt{g}} \cdot ((\iota - \partial_\zeta \lambda) \mathbf{e}_\theta + (1 + \partial_\theta \lambda) \mathbf{e}_\zeta)

The current density is then calculated from Ampere's Law, :math:`\nabla \times \mathbf{B} = \mu_0 \mathbf{J}`, 

.. math::
  \begin{aligned}
  J^\rho &= \frac{\partial_\theta B_\zeta - \partial_\zeta B_\theta}{\mu_0 \sqrt{g}} \\
  J^\theta &= \frac{\partial_\zeta B_\rho - \partial_\rho B_\zeta}{\mu_0 \sqrt{g}} \\
  J^\zeta &= \frac{\partial_\rho B_\theta - \partial_\theta B_\rho}{\mu_0 \sqrt{g}}
  \end{aligned}

where :math:`B_i = \mathbf{B} \cdot \mathbf{e}_i`. 
This allows the magnetic field and current density to be computed from the independent variables and inputs: 

.. math::
  \begin{aligned}
  \mathbf{B}(\rho, \theta, \zeta) &= \mathbf{B}(R(\rho, \theta, \zeta), Z(\rho, \theta, \zeta), \lambda(\rho, \theta, \zeta), \iota(\rho)) \\
  \mathbf{J}(\rho, \theta, \zeta) &= \mathbf{J}(R(\rho, \theta, \zeta), Z(\rho, \theta, \zeta), \lambda(\rho, \theta, \zeta), \iota(\rho))
  \end{aligned}

Equilibrium Force Balance
*************************

The ideal magnetohydrodynamic equilibrium force balance is defined as 

.. math::
  \mathbf{F} \equiv \mathbf{J} \times \mathbf{B} - \nabla p = \mathbf{0}

When written in flux coordinates there are only two independent components: 

.. math::
  \begin{aligned}
  \mathbf{F} &= F_\rho \nabla \rho + F_\beta \mathbf{\beta} \\
  F_\rho &= \sqrt{g} (B^\zeta J^\theta - B^\theta J^\zeta) - \partial_\rho p \\
  F_\beta &= \sqrt{g} B^\zeta J^\rho \\
  \mathbf{\beta} &= \nabla \theta - \iota \nabla \zeta
  \end{aligned}

These forces in both the radial and helical directions must vanish in equilibrium. 
DESC solves this force balance locally by evaluating the residual errors at discrete points in real space: 

.. math::
  \begin{aligned}
  f_\rho = F_\rho ||\nabla \rho|| \Delta V \\
  f_\beta = F_\beta ||\mathbf{\beta}|| \Delta V
  \end{aligned}

These equations :math:`f_\rho` and :math:`f_\beta` represent the force errors (in Newtons) in the unit of volume :math:`\Delta V = \sqrt{g} \Delta \rho \Delta \theta \Delta \zeta` surrounding a collocation point :math:`(\rho, \theta, \zeta)`. 
[Note: this definition of :math:`\mathbf{\beta}` is slightly different from that given in the original paper, but the resulting equation for :math:`f_\beta` is equivalent. 
The publication also included an additional sign term in the equations for :math:`f_\rho` and :math:`f_\beta` that has been dropped.] 

In summary, the equilibrium problem is formulated as a system of nonlinear equations :math:`\mathbf{f}(\mathbf{x}, \mathbf{c}) = \mathbf{0}`. 
The state vector :math:`\mathbf{x}` contains the spectral coefficients representing the independent variables: 

.. math::
  \mathbf{x} = [R_{lmn} \\ Z_{lmn} \\ \lambda_{lmn}]^T

The parameter vector :math:`\mathbf{c}` contains the spectral coefficients of the inputs that define a unique equilibrium solution: 

.. math::
  \mathbf{c} = [R^b_{mn} \\ Z^b_{mn} \\ p_l \\ \iota_l \\ \psi_a]^T

The equations :math:`\mathbf{f}` are the force error residuals at a series of collocation points, as well as additional equations to enforce the boundary condition: 

.. math::
  \mathbf{f} = [f_\rho \\ f_\beta \\ BC]^T

DESC allows flexibility in the choice of optimization algorithm used to solve this system of equations; popular approaches include Newton-Raphson methods and least-squares minimization. 
