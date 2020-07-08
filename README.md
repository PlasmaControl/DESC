# DESC
Stellarator Equilibrium Solver

This is a lightweight version of the DESC code in MATLAB.  No external dependencies are required, and the code should run from any platform.  Testing was performed in MATLAB r2019b.  

Inputs must be provided in an input file that is referenced by `runner.m`.  See `input_Dshape.m` and `input_Heliotron.m` for examples and documentation.  To calculate an equilibrium, simply run the program `runner.m`.  The equilibrium solution is saved in the file `xequil.mat`.  Run the program `plotter.m` to visualize the equilibrium flux surfaces and force balance errors.  