%% plots equilibrium solution

input_Dshape
% input_Heliotron

load('xequil.mat','xequil')
plot_x(xequil,bndryR,bndryZ,NFP,M(end),N(end),symm)
plot_f(xequil,Pres,Iota,Psi,bndryR,bndryZ,NFP,M(end),N(end),symm)
