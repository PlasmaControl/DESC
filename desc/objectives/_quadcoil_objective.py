from desc.objectives.objective_funs import _Objective, collect_docs
from desc.magnetic_fields import QuadcoilThing

class QuadcoilObjective(_Objective):
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
        qt:QuadcoilThing,
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
                '(See the API for QuadcoilThing) Any non-default values '
                'of normalize and normalize_target will be overridden.')

        self.f_quadcoil = qt.f_quadcoil
        self._static_attrs = [
            'f_quadcoil',
        ]
        
        # ----- Superclass -----
        super().__init__(
            things=[qt.eq, qt], # things is a list of things that will be optimized, in this case just the equilibrium
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=False,
            normalize_target=False,
            name=name,
            jac_chunk_size=jac_chunk_size
        )
    
    def compute(self, params_eq, params_qt):
        qt = self.things[1]
        qp = qt.params_to_qp(params_eq, params_qt)
        dofs = qt.params_to_dofs(params_qt)
        return self.f_quadcoil(qp, dofs)
