%% plots equilibrium solution

clearvars

input_Dshape

load('xequil.mat','xequil')
plot_x(xequil,bndryR,bndryZ,NFP,M(end),N(end),lm,ln,symm,squr)
plot_f(xequil,Pres,Iota,Psi,bndryR,bndryZ,NFP,M(end),N(end),lm,ln,symm,squr)
