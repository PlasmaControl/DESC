This folder contains the files necessary to reproduce the figures in the paper "The DESC Stellarator Code Suite Part III: Quasi-symmetry optimization".
A copy of this paper is included as `dudt2022optimization.pdf`.

In order to run the following code, checkout version `0fcc708` of desc.

`driver.py` is a Python script that performs the optimizations in Section III B.
There are two optimization parameters that can be set in lines 46 and 47 of the script.
`qs` is a character that sets the quasi-symmetry objective function, and can be either "B"`, `"C"`, or `"T"`.
These correspond to the options detailed in Section II C.
`order` is an integer that specifies the order of the perturbation used, according to the theory explained in Section II B. It can be either `1` or `2`.

`plotter.py` is a Python script that generates the plots shown in Figures 1-5.
The following plots are all saved in the `data` sub-folder:
- `f_B.png` corresponds to Figure 1
- `f_C.png` corresponds to Figure 2
- `f_T.png` corresponds to Figure 3
- `Booz.png` corresponds to Figure 4
- `errors.png` corresponds to Figure 5

The `data` sub-folder also contains the following data used to generate the plots:
- `initial_input` is the DESC input file used to generate the initial equilibrium solution.
- `initial.h5` is the Equilibrium class object representing the initial guess used for optimization.
- `eq_fB_or1.h5` etc. are Equilibrium objects corresponding to the six different DESC optimal solutions.
- `stellopt.h5` is an Equilibrium object representing the equivalent STELLOPT solution.
- `f_B.npy`, `f_C.npy`, and `f_T.npy` are NumPy arrays representing the optimization ladscape shown in the shaded background of Figures 1-3.
- `rbc_fB_or1.npy` and `zbs_fB_or1.npy` etc. are NumPy arrays corresponding to the six different DESC optimization paths.
- `rbc_stellopt.npy` and `zbs_stellopt.npy` are NumPy arrays representing the STELLOPT optimization path.
