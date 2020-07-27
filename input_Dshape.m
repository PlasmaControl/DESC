%% input file for D-shaped tokamak beta=3%

%% solver

% symm = true for stellarator symmetry
% squr = true for square system

symm = true;
squr = false;

% M = maximum poloidal Fourier mode
% N = maximum toroidal Fourier mode
% max radial resolution = rho^{2M}
% resolutions are given as an vector for continuation method

M = 6:12;
N = zeros(size(M));

% options = fsolve algorithm options

options = optimoptions('fsolve','Display','iter-detailed',...
    'Algorithm','levenberg-marquardt','SpecifyObjectiveGradient',false,...
    'MaxFunctionEvaluations',1e8,'MaxIterations',1e4,...
    'FunctionTolerance',1e-6,'StepTolerance',1e-4);

%% boundary

% the plasma boundary is defined by a Fourier series of the form:
% R = R_{mn}*sin(|m|*theta)*sin(|n|*NFP*phi)  for m < 0,  n < 0
%   + R_{mn}*sin(|m|*theta)*cos(|n|*NFP*phi)  for m < 0,  n >= 0
%   + R_{mn}*cos(|m|*theta)*sin(|n|*NFP*phi)  for m >= 0, n < 0
%   + R_{mn}*cos(|m|*theta)*cos(|n|*NFP*phi)  for m >= 0, n >= 0
% same for Z

% (R,phi,Z) is a right-handed coordinate system
% theta is measured clockwise from the inboard midplane

% boundary Fourier components are given in a matrix:
% rows range from -M to +M, columns range from -N to +N
% bndryR(m+M+1,n+N+1) = R_{mn}
% bndryZ(m+M+1,n+N+1) = Z_{mn}

% resolution for boundary can be less than total resolution (M,N)

bndryR = [0.000; 0.00; 3.51; -1.00; 0.106];
bndryZ = [0.160; 1.47; 0.00;  0.00; 0.000];
[lm,ln] = size(bndryR);
bndryR = bndryR(:);
bndryZ = bndryZ(:);

% NFP = number of field periods

NFP = 1;

%% profiles

% Pres = coefficients of pressure profile
% Iota = coefficients of rotational transform profile

% coefficients correspond to the purely radial (m=0) Zernike polynomials
% the basis functions are: 1, 2*rho^2-1, 6*rho^4-6*rho^2+1, ...

p0 = 1.65e3;
Pres = p0*[1/3; -1/2; 1/6];
Iota = [1-0.67/2; -0.67/2];

% Psi = toroidal magnetic flux at plasma boundary (Wb)

Psi = 1;
