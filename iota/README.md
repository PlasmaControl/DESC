Note that the symmetry and NFP bugs in the grid class described in magwell/magwell-test.html persist. My solution for the NFP bug is still in that notebook. Until this is fixed, any quantity that relies on `grid.spacing` (except for their triple product, `grid.weights`) will give incorrect results under the conditions described in that notebook.

I've added a unit test in tests/test_compute_utils.py that may help debug this. The function is `test_surface_area_unweighted`. Currently it passes when the supplied grid is `random_grid(NFP=1, sym=False)`. When the sym/NFP issues are fixed, I think this test should pass when `NFP != 1` and `sym=True`.

For testing the rotational transform we need to make sure we
* compute the rotational transform on grids with NFP = 1 and sym = False.
* when the input file has current coefficients, we need to solve equilibria with grids with NFP = 1 and sym = False because iota is used in the force balance objective. So make sure to change equilibrium file inputs to reflect this.
