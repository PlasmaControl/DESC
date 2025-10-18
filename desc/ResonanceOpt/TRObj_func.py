from desc.objectives import (
    TrappedResonance
)
import numpy as np

# Run objective function
def TrappedResonanceObj(eq,rhos,pitch_invs,KE_frac,alphas,N):
    obj = TrappedResonance(eq,rho=rhos,pitch_invs=pitch_invs,KE_frac=KE_frac,alpha=alphas,N=N)
    obj.build()
    out = obj.compute(eq.params_dict) # when not flattened, this shape is (rho,pitch,energy)

    # Save objective function values to the firm3d directory for plotting with Poincare plots
    # np.savetxt('/Users/paullab/codes/firm3d_fork_10132025/firm3d/examples/trapped_map/obj_DESC.txt',out['obj'][:,0,0]) # value at all surfaces and one pitch (and one energy)
    return out