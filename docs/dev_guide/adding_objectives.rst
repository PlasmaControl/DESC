==============================
Adding new objective functions
==============================

.. attention::
    This page is mainly intended to explain some of the logic inside of objective functions.
    For simple objectives like shown in this page, it is recommended to use the [``GenericObjective``](https://desc-docs.readthedocs.io/en/latest/_api/objectives/desc.objectives.GenericObjective.html#desc.objectives.GenericObjective)
    (for objectives that simply just use values computable already in the data index, see
    [List of Variables](https://desc-docs.readthedocs.io/en/latest/variables.html))
    or [``ObjectiveFromUser``](https://desc-docs.readthedocs.io/en/latest/_api/objectives/desc.objectives.ObjectiveFromUser.html#desc-objectives-objectivefromuser)
    (for quantities which are derived from things computable from the data index)
    classes. The benefit of making a full objective class like shown in this page is mainly when dealing
    with multiple objects at once (see e.g [``PlasmaVesselObjective``](https://desc-docs.readthedocs.io/en/latest/_api/objectives/desc.objectives.PlasmaVesselDistance.html#desc.objectives.PlasmaVesselDistance)),
    or more complicated objectives (like [``EffectiveRipple``](https://desc-docs.readthedocs.io/en/latest/_api/objectives/desc.objectives.EffectiveRipple.html#desc.objectives.EffectiveRipple))
    or having better control over default values and overriding some methods such as ``print_value``.
    Most objectives can trivially be made with ``GenericObjective`` and ``ObjectiveFromUser``:
    ::

        from desc.objectives import ObjectiveFromUser, GenericObjective
        from desc.equilibrium import Equilibrium

        eq = Equilibrium()

        # QS triple product, already exists as "f_T" in the data index
        obj_QS_triple = GenericObjective(f="f_T", thing=eq)

        # Mirror ratio, manually computed from "|B|"
        def fun_mirror_ratio(grid, data):
            max_tz_B = surface_max(grid=grid, x=data["|B|"], surface_label="rho")
            min_tz_B = surface_min(grid=grid, x=data["|B|"], surface_label="rho")

            max_tz_B = grid.compress(max_tz_B, surface_label="rho")
            min_tz_B = grid.compress(min_tz_B, surface_label="rho")

            mirror_ratio = (max_tz_B - min_tz_B) / (min_tz_B + max_tz_B)

            return mirror_ratio
            # alternatively, "mirror ratio" is something that can be computed in the data index
            # directly (see List of Variables docs), so can replace entirety of the above function code with this return statement
            # return grid.compress(data["mirror ratio"])
            # or can just use GenericObjective(f="mirror ratio", thing=eq) like given above

        obj_mirror_ratio = ObjectiveFromUser(fun=fun_mirror_ratio, thing=eq)

This guide walks through creating a new objective to optimize using Quasi-symmetry as
an example. The primary methods needed for a new objective are ``__init__``, ``build``,
and ``compute``. The base class ``_Objective`` provides a number of other methods that
generally do not need to be re-implemented for subclasses such as derivative calculation,
deviation from the target value, scaled cost and many common attributes.

``__init__`` should generally just assign attributes and store inputs. The object(s) that will be optimized by
the objective is (are) also defined in the ``__init__`` method. This (these) can be any object that is inherited from
``Optimizable`` super-class, i.e. ``Equilibrium``, ``Coil``, ``Surface``. ``__init__`` method should not do
any expensive calculations, these should be in ``build`` or ``compute``. The main
arguments are summarized in the example below.

``build`` is called before optimization either by user or automatically. It is used to
precompute things that will be constant during the optimization like transform matrices that convert spectral
coefficients to real space values, the values of the fixed plasma profiles, names of the physics quantities
required that are registered as compute functions, shape of the objective value etc.
The quantities that will be used by ``compute`` are then packaged into ``constants``. ``build`` method
also performs any necessary checks on the inputs before starting the optimization.

``compute`` is where the actual calculation of the objective takes place. Objectives
generally return a vector of residuals that are minimized in a least squares sense,
though the exact method will depend on the optimization algorithm. The main thing here
is calling ``compute_fun`` to get physics quantities, and then performing any
post-processing we want such as averaging, combining, etc.

Now let's look at ``QuasisymmetryTripleProduct`` as an example. This objective takes in an ``Equilibrium``
object and computes the quasi-symmetry triple product.

First, we need some common imports that almost all objectives in DESC need. One thing to keep in mind here is that ``desc.compute.utils``
offers 2 functions namely ``compute`` and ``_compute``. The latter is the JIT compatible version of the former, and is used
in the ``compute`` method of the objective. The former has additional checks that make it incompatible to use in JIT-compiled
functions. We import ``_compute`` with an alias ``compute_fun`` to avoid confusion with the ``compute`` method of the objective.
::

    from desc.objectives.objective_funs import _Objective
    from desc.objectives.normalization import compute_scaling_factors
    from desc.compute import get_profiles, get_transforms
    from desc.compute.utils import _compute as compute_fun


``_Objective`` parent class provides a lot of functionality for objectives, such as ``compute_scaled`` for scaling the result of
``compute`` method, ``compute_scaled_error`` for computing the error from the target, ``jac_scaled_error`` for computing the
Jacobian of the scaled error, and many other methods. The docstring of the objective should explain special inputs. The docstrings
for common inputs (i.e. ``target``, ``bounds``, ``weight``, ``jac_chunk_size`` etc)  can be inherited from the base class
and can be adjusted as shown below.
::

    class QuasisymmetryTripleProduct(_Objective):  # need to subclass from ``desc.objectives._Objective``
        """Give a description of what it is and what it's useful for.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium that will be optimized to satisfy the Objective.
        grid : Grid, optional
            Collocation grid containing the nodes to evaluate at.

        """
        # Most of the documentation is shared among all objectives, so we just inherit
        # the docstring from the base class and add a few details specific to this objective.
        # See the documentation of `collect_docs` for more details.
        __doc__ = __doc__.rstrip() + collect_docs(
            target_default="``target=0``.", bounds_default="``target=0``."
        )

        _coordinates = "rtz"    # What coordinates is this objective a function of, with r=rho, t=theta, z=zeta?
                                # i.e. if only a profile, it is "r" , while if all 3 coordinates it is "rtz"
        _units = "(T^4/m^2)"    # units of the output
        _print_value_fmt = "Quasi-symmetry error: "    # string with the name of the printed value, used when showing results of an optimization


``__init__`` method should assign the optimizable thing(s) to the ``things`` attribute, which is a list of objects
that will be optimized. For this example, we will optimize an ``Equilibrium`` object, so we assign it to the
``things`` as a list. As explained before, the ``__init__`` method should not do any expensive calculations, so we just assign the
attributes and call the parent class's ``__init__`` method which will handle common inputs and finalize the initialization.
::

        def __init__(
            self,
            eq,
            target=None,
            bounds=None,
            weight=1,
            normalize=True,
            normalize_target=True,
            grid=None,
            name="QS triple product",
            jac_chunk_size=None,
        ):
            # we don't have to do much here, mostly just call ``super().__init__()``
            if target is None and bounds is None:
                target = 0 # default target value
            self._grid = grid
            super().__init__(
                things=[eq], # things is a list of things that will be optimized, in this case just the equilibrium
                target=target,
                bounds=bounds,
                weight=weight,
                normalize=normalize,
                normalize_target=normalize_target,
                name=name,
                jac_chunk_size=jac_chunk_size
            )

``build`` method can be thought as a pre-computation step that prepares the objective for optimization by storing the constants
needed for ``compute`` method to prevent extra computations. This method is not JIT-compiled, so it can perform any Python code.

``grid`` is a ``Grid`` object that contains the nodes where the objective will be evaluated. If it is not provided, a default
grid is created based on the grid requirements for the objective. For example, if the objective needs to compute a volumetric
quantity, a grid that covers the entire plasma volume needs to be chosen as default, or if there is an integral quantity
a grid with proper quadrature points needs to be chosen. Sometimes 2 grids are needed, for example coil objectives, one for the
evaluation points on plasma surface and one for the coil segments for Biot-Savart integration.

Probably the most important part of the ``build`` method is to call ``get_profiles`` and ``get_transforms`` functions
from ``desc.compute.utils``. These functions return the profiles and transforms needed to compute the physics
quantities from the equilibrium object. Both functions return dictionaries. Since these require information on the
computation grid, one needs to call them after assigning the grid to the objective.

``_data_keys`` is a list of strings that specifies which physics quantities are needed
to be computed, for this example, from the equilibrium object. If there are multiple things in ``self.things``, one
can create separate lists for each thing. One can use a different name instead of ``_data_keys``, but it is a convention
in most DESC objectives. ``_dim_f`` is the size of the output vector returned by ``compute`` method.
This quantity is used in ``ObjectiveFunction`` class to conduct concatenation or splitting and the name
``_dim_f`` has to be kept to prevent errors. One should also define the proper normalization factor for
the objective, if needed. The units of the normalization factor should be such that the objective value is unitless.

We put all the constants into a dictionary called ``self._constants``. This dictionary will be passed to the
``compute`` method as the ``constants`` argument, so it can access the transforms and profiles needed to compute the objective.
Alternatively, one can also store the constants as attributes of the objective, for instance ``self._transforms``
and ``self._profiles``. Finally, we call the parent class's ``build`` method for common parts of building the objective.
::

        def build(self, use_jit=True, verbose=1):
            """Build constant arrays.

            Parameters
            ----------
            use_jit : bool, optional
                Whether to just-in-time compile the objective and derivatives.
            verbose : int, optional
                Level of output.

            """
            # things is the list of things that will be optimized,
            # we assigned things to be just eq in the init, so we know that the
            # first (and only) element of things is the equilibrium
            eq = self.things[0]
            # need some sensible default grid
            if self._grid is None:
                grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
            else:
                grid = self._grid
            # dim_f = size of the output vector returned by self.compute
            # usually the same as self.grid.num_nodes, unless you're doing some down-sampling
            # or averaging etc.
            self._dim_f = self.grid.num_nodes
            # What data from desc.compute is needed? Here we want the QS triple product.
            self._data_keys = ["f_T"]

            # some helper code for profiling and logging
            timer = Timer()
            if verbose > 0:
                print("Precomputing transforms")
            timer.start("Precomputing transforms")

            # helper functions for building transforms etc to compute given
            # quantities. Alternatively, these can be created manually based on the
            # equilibrium, though in most cases that isn't necessary.
            profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
            transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
            self._constants = {
                "transforms": transforms,
                "profiles": profiles,
            }

            timer.stop("Precomputing transforms")
            if verbose > 1:
                timer.disp("Precomputing transforms")


            # We try to normalize things to order(1) by dividing things by some
            # characteristic scale for a given quantity.
            # See ``desc.objectives.compute_scaling_factors`` for examples.
            if self._normalize:
                scales = compute_scaling_factors(eq)
                # since the objective has units of T^4/m^2, the normalization here is
                # based on a characteristic field strength and minor radius.
                self._normalization = (
                    scales["B"] ** 4 / scales["a"] ** 2
                )

            # finally, call ``super.build()``
            super().build(use_jit=use_jit, verbose=verbose)


The actual computation of the objective happens in ``compute`` method. This method is JIT-compiled
(unless ``use_jit=False`` is passed to ``build`` method), so it should only contain JIT-compatible code.
This method takes in the parameters of the thing(s) to be optimized, which is the dictionary form of the
state vector such as `R_lmn`, `Z_lmn`, etc. for the ``Equilibrium`` object. Objectives with multiple ``things``
can have multiple parameters, one for each thing in ``self.things``, in this case, the function signature would be
``compute(self, params_1, params_2, params3, ..., constants=None)``, see the
[``PlasmaVesselDistance``](https://desc-docs.readthedocs.io/en/latest/_api/objectives/desc.objectives.PlasmaVesselDistance.html#desc.objectives.PlasmaVesselDistance)
objective for an example of this. The ``constants`` argument is a dictionary of any other constant and usually set to ``None``
so that the ``self.constants`` are used.
::

        def compute(self, params, constants=None):
            """Signature should take params (or possibly multiple params, one for each thing in self.things),
               which is the params_dict of the expected thing(s) to be optimized.
               It also takes in constants, which is a dictionary of any other constant data needed to compute
               the objective, and is usually none by default so the self.constants are used.

            Parameters
            ----------
            params : dict
                Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
            constants : dict
                Dictionary of constant data, eg transforms, profiles etc. Defaults to
                self.constants

            Returns
            -------
            f : ndarray
                Quasi-symmetry flux function error at each node (T^4/m^2).

            """
            if constants is None:
                constants = self.constants

            # here we get the physics quantities from ``desc.compute.utils._compute``
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._data_keys,                 # quantities we want
                params=params,                   # params from input containing the equilibrium R_lmn, Z_lmn, etc
                transforms=self._transforms,     # transforms and profiles from self.build
                profiles=self._profiles,
            )
            # next we do any additional processing, such as combining things,
            # averaging, etc. Here we just return the QS triple product f_T evaluated at each
            # node in the grid.
            f = data["f_T"]
            # this is all we need to do here. Applying objective weights/targets/bounds
            # is handled by the base _Objective class, as well as the normalizations to be unitless
            # and to make the objective value independent of grid resolution.
            return f

Converting to Cartesian coordinates
-----------------------------------

The above example of quasi-symmetry is a scalar quantity that is independent of the
coordinate system. ``desc.compute.utils._compute`` always returns all vector quantities
in toroidal coordinates :math:`(R,\phi,Z)`. If you would prefer to work in Cartesian
coordinates :math:`(X,Y,Z)` for any intermediate computations within your new objective,
you will have to manually convert these vectors using the geometry utility functions
``rpz2xyz`` and/or ``rpz2xyz_vec``. See the ``PlasmaVesselDistance`` objective for an
example of this.

Adapting Existing Objectives with Different Loss Functions
----------------------------------------------------------

If your desired objective is already implemented in DESC, but not in the correct form,
a few different loss functions are available through the ``loss_function`` kwarg
when instantiating an Objective, to modify the objective cost in order to adapt
the objective to your desired purpose. For example, the DESC ``RotationalTransform``
objective with ``target=iota_target`` by default forms the residual by taking the target
and subtracting it from the profile at the points in the grid, resulting in a residual
of the form :math:`\iota_{err} = \sum_{i} (\iota_i - iota_{target})^2`, i.e. the residual
is the sum of squared pointwise error between the current rotational transform profile
and the target passed into the objective. If the desired objective instead is to
optimize to target an average rotational transform of `iota_target`, we can adapt the
``RotationalTransform`` object by passing in ``loss_function="mean"``. The options
available for the ``loss_function`` kwarg are ``[None,"mean","min","max"]``, with
``None`` meaning using the usual default objective cost, while ``"mean"`` takes the
average of the raw objective values (before subtracting the target/bounds or
normalization), ``"min"`` takes the minimum, and ``"max"`` takes the maximum.
