## Grid.py bugs status

With the new updates to the grid class and my NFP bug fix,
* correct computations result on these grids
    - $\forall g \in Grid : g.num\_rho \in \mathbb{1}\ and\ g.NFP \in \mathbb{R}^{+}\ and\ g.sym \in \mathbb{True, False}$.
* incorrect computations result on these grids
    - $\forall g \in Grid : g.num\_rho \in \mathbb{N} \setminus {1}\ and\ g.NFP \in \mathbb{R}^{+} \setminus {1}$.
        - I have a solution for this, see above cell. But we should discuss before I implement it.
    - $\forall g \in Grid : g.num\_rho \in \mathbb{N} \setminus {1}\ and\ g.NFP \in \mathbb{1}\ and\ g.sym \in \mathbb{True}$.

## Inputs are identical to those in examples/DESC except that
* sym is set to 0
* NFP is set to 1
* increased nfev
* current profile computed from fitting the current computed from a fixed-iota solved equilibrium. See notebook.

### pass list
* DShape
* Solovev

### fail list, see ./logs.
* Heliotron: "Killed" message
* ESTELL: "Killed" message.
* WISTELL-A: Flux surfaces are no longer nested.
WISTELL-A and ESTELL are vacuum cases. I know this is a bad test for tokamaks, but I think it's ok for stellarators.

## Small bugs
When using DESC to automatically convert a VMEC input to DESC input,
* DESC forgets to scale current profile coefficients by VMEC's CURTOR scaling factor.
* When symmetry is off in VMEC input, DESC converts this to `sym = False` in the DESC input file. The resulting input file will not run. DESC should instead convert this to `sym = 0`.
* keep seeing "RuntimeWarning: Save attribute '_iota' was not saved as it does not exist." in fix-current mode and vice verse.
