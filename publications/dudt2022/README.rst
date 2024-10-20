This folder contains the files necessary to reproduce the figures in the paper "The DESC Stellarator Code Suite Part III: Quasi-symmetry optimization".
A copy of this paper is included as `dudt2022optimization.pdf`.

`driver.py` is a Python file that performs the optimizations in Section 3.2 of the paper.
In order to run this script, checkout version `0fcc708` of the DESC code.
There are two optimization parameters that can be set in lines 48 and 49 of the script.
`qs` is a character that sets the quasi-symmetry objective function, and can be either `"B"`, `"C"`, or `"T"`.
These correspond to the options detailed in Section 2.3 of the paper.
`order` is an integer that specifies the order of the perturbation used, and can be either `1` or `2`.
This corresponds to the theory explained in Section 2.2 of the paper.

`plotter.py` is a Python file that generates the plots shown in Figures 1-6.
In order to run this script, checkout version `v0.7.2` of the DESC code.
The following plots are all saved in the `data` sub-folder:
- `f_B.png` corresponds to Figure 1
- `f_C.png` corresponds to Figure 2
- `f_T.png` corresponds to Figure 3
- `boundaries.png` corresponds to Figure 4
- `Booz.png` corresponds to Figure 5
- `errors.png` corresponds to Figure 6

The `data` sub-folder also contains the following data used to generate the plots:
- `initial_input` is the DESC input file used to generate the initial equilibrium solution.
- `initial.h5` is the Equilibrium class object representing the initial guess used for optimization.
- `eq_fB_or1.h5` etc. are Equilibrium objects corresponding to the six different DESC optimal solutions.
- `stellopt.h5` is an Equilibrium object representing the equivalent STELLOPT solution.
- `f_B.npy`, `f_C.npy`, and `f_T.npy` are NumPy arrays representing the optimization ladscape shown in the shaded background of Figures 1-3.
- `rbc_fB_or1.npy` and `zbs_fB_or1.npy` etc. are NumPy arrays corresponding to the six different DESC optimization paths.
- `rbc_stellopt.npy` and `zbs_stellopt.npy` are NumPy arrays representing the STELLOPT optimization path.
