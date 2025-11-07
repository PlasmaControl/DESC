from desc.objectives.objective_funs import _Objective, collect_docs
from desc.backend import jnp
from jax import eval_shape
import jax
from jax import disable_jit

class QuadcoilConstraint(_Objective):
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
    _print_value_fmt = "QUADCOIL constraint: "  
    def __init__(
        self,
        qf,
        target=None, # None,
        bounds=(-jnp.inf, 0.),
        weight=1.,
        normalize=None,
        normalize_target=None,
        name='QUADCOIL constraint',
        jac_chunk_size=None,
    ):
        if normalize is not None or normalize_target is not None:
            raise AttributeError(
                'QUADCOIL performs its own normalization. '
                '(See the API for QuadcoilField) Any non-default values '
                'of normalize and normalize_target will be overridden.')
        
        self._static_attrs = _Objective._static_attrs + [
            '_static_attrs', 
            '_deriv_mode',
            '_verbose',
        ]

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
        eq = self.things[0]
        qf = self.things[1]
        # 1:00 issue is between here
        # dim_f = size of the output vector returned by self.compute.
        # We now count the total number of scalar constraints in the 
        # problem.
        dim_g = eval_shape(
            qf._g_quadcoil, 
            qf.params_to_qp(eq.params_dict, qf.params_dict),
            qf.params_to_dofs(qf.params_dict)
        ).size
        dim_h = eval_shape(
            qf._h_quadcoil, 
            qf.params_to_qp(eq.params_dict, qf.params_dict),
            qf.params_to_dofs(qf.params_dict)
        ).size
        self._dim_f = dim_g + dim_h
        # 1:00 issue is between here

        # params_eq = eq.params_dict
        # params_qf = qf.params_dict
        # qp = qf.params_to_qp(params_eq, params_qf)
        # dofs = qf.params_to_dofs(params_qf)
        # g_dummy = qf._g_quadcoil(qp, dofs)
        # h_dummy = qf._h_quadcoil(qp, dofs)
        # self._dim_f = g_dummy.size + h_dummy.size
        # 14:02: using 1 constraints is fast but 2177 isn't 
        # is it just inefficient for high constraint #?
        # self._dim_f = 2000# 2177 # TESTING
        super().build(use_jit=use_jit, verbose=verbose)
    
    def compute(self, *all_params, constants=None):
        params_eq = all_params[0]
        params_qf = all_params[1]
        qf = self.things[1]
        # All logics are packcaged into quadcoilfields
        qp = qf.params_to_qp(params_eq, params_qf)
        dofs = qf.params_to_dofs(params_qf)
        g_vals = qf._g_quadcoil(qp, dofs)
        h_vals = qf._h_quadcoil(qp, dofs)
        return jnp.concatenate([g_vals,h_vals])
