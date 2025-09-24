from desc.objectives.objective_funs import _Objective, collect_docs

class QuadcoilObjective(_Objective):
    """
    Dummy
    """
    # Most of the documentation is shared among all objectives, so we just inherit
    # the docstring from the base class and add a few details specific to this objective.
    # See the documentation of `collect_docs` for more details.
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )
    _coordinates = ""    # What coordinates is this objective a function of, with r=rho, t=theta, z=zeta?
                            # i.e. if only a profile, it is "r" , while if all 3 coordinates it is "rtz"
    _units = "N/A"    # units of the output
    _print_value_fmt = "QUADCOIL objective: "    # string with python string formatting for printing the value
    def __init__(
        self,
        qf,
        target=0,
        bounds=None,
        weight=1.,
        normalize=None,
        normalize_target=None,
        name='QUADCOIL objective',
        jac_chunk_size=None,
    ):
        if normalize is not None or normalize_target is not None:
            raise AttributeError(
                'QUADCOIL performs its own normalization. '
                '(See the API for QuadcoilField) Any non-default values '
                'of normalize and normalize_target will be overridden.')

        # self._f_quadcoil = qf._f_quadcoil
        # self._static_attrs = [
        #     'f_quadcoil',
        # ]
        
        # ----- Superclass -----
        super().__init__(
            things=[qf.eq, qf], # things is a list of things that will be optimized, in this case just the equilibrium
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=False,
            normalize_target=False,
            name=name,
            jac_chunk_size=jac_chunk_size
        )

    def build(self, use_jit=True, verbose=1):
        # Nothing needed here.
        # All of the logics are packaged into 
        # QuadcoilField.

        # dim_f = size of the output vector returned by self.compute.
        # This is a scalar objective.
        self._dim_f = 1
        super().build(use_jit=use_jit, verbose=verbose)
    
    def compute(self, params_eq, params_qf):
        qf = self.things[1]
        qp = qf.params_to_qp(params_eq, params_qf)
        dofs = qf.params_to_dofs(params_qf)
        return qf._f_quadcoil(qp, dofs)
