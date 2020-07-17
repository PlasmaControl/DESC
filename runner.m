%% calculates equilibrium solution
function [] = runner()

symm = true;
squr = true;

% M = maximum poloidal resolution (max radial resolution = rho^{2M})
% N = maximum toroidal resolution
% resolutions are given as an vector for continuation method

M = 6:6;
N = zeros(size(M));

% options = fsolve algorithm options

atol = 1e-6;
rtol = 1e-6;
steptol = 1e-8;
max_iter = 1e4;
h = 1e-4;
options = optimoptions('fsolve','Display','iter-detailed',...
    'Algorithm','levenberg-marquardt','SpecifyObjectiveGradient',false,...
    'MaxFunctionEvaluations',1e8,'MaxIterations',max_iter,...
    'FunctionTolerance',rtol,'StepTolerance',steptol);

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



t1=tic;
% input_Dshape
% input_Heliotron

for k = 1:length(M)
    
%     fprintf('\nM = %u\nN = %u\n',M(k),N(k))
    if k == 1  % scaled boundary initial guess
        xequil = x_init(bndryR,bndryZ,NFP,M(k),N(k),symm);
    else       % continue from lower resolution solution
%         load('xequil.mat','xequil');
        xequil = expandx(xequil,M(k-1),N(k-1),M(k),N(k),symm);
    end
    
    if squr  % square system
        Mnodes = M(k);
    else     % oversample by 3/2
        Mnodes = ceil(1.5*M(k));
    end
    % collocation nodes
    [rC0,vC0,rS0,vS0,rC1,vC1,rS1,vS1,drC0,dvC0,drS0,dvS0,drC1,dvC1,drS1,dvS1] = nodes(Mnodes);
    % pre-compute interpolation matrices
    [iM,ZERN_C0,ZERNr_C0,ZERNv_C0,ZERNrr_C0,ZERNvv_C0,ZERNrv_C0,ZERNrrv_C0,ZERNrvv_C0,ZERNrrvv_C0] = zernfun(M(k),rC0,vC0);
    [~ ,ZERN_S0,ZERNr_S0,ZERNv_S0,ZERNrr_S0,ZERNvv_S0,ZERNrv_S0,ZERNrrv_S0,ZERNrvv_S0,ZERNrrvv_S0] = zernfun(M(k),rS0,vS0);
    [~ ,ZERN_C1,ZERNr_C1,ZERNv_C1,ZERNrr_C1,ZERNvv_C1,ZERNrv_C1,ZERNrrv_C1,ZERNrvv_C1,ZERNrrvv_C1] = zernfun(M(k),rC1,vC1);
    [~ ,ZERN_S1,ZERNr_S1,ZERNv_S1,ZERNrr_S1,ZERNvv_S1,ZERNrv_S1,ZERNrrv_S1,ZERNrvv_S1,ZERNrrvv_S1] = zernfun(M(k),rS1,vS1);
    % profile Zernike coefficients - stick them into the m=0 azimuthally
    % symmetric coeffs
    cP = zeros((M(k)+1)^2,1);  cP(find(iM==0,length(Pres))) = Pres;
    cI = zeros((M(k)+1)^2,1);  cI(find(iM==0,length(Iota))) = Iota;
    
    fun = @(x) f_err(x,cP,cI,Psi,bndryR,bndryZ,NFP,M(k),N(k),iM,...
        ZERN_C0,ZERNr_C0,ZERNv_C0,ZERNrr_C0,ZERNvv_C0,ZERNrv_C0,ZERNrrv_C0,ZERNrvv_C0,ZERNrrvv_C0,...
        ZERN_S0,ZERNr_S0,ZERNv_S0,ZERNrr_S0,ZERNvv_S0,ZERNrv_S0,ZERNrrv_S0,ZERNrvv_S0,ZERNrrvv_S0,...
        ZERN_C1,ZERNr_C1,ZERNv_C1,ZERNrr_C1,ZERNvv_C1,ZERNrv_C1,ZERNrrv_C1,ZERNrvv_C1,ZERNrrvv_C1,...
        ZERN_S1,ZERNr_S1,ZERNv_S1,ZERNrr_S1,ZERNvv_S1,ZERNrv_S1,ZERNrrv_S1,ZERNrvv_S1,ZERNrrvv_S1,...
        rC0,drC0,dvC0,rS0,drS0,dvS0,rC1,drC1,dvC1,rS1,drS1,dvS1,symm,squr);
    % solve system of equations and save equilibrium
    t2=tic;
%     xequil = myfsolve(fun,xequil,rtol,steptol,max_iter,h,true,'homotopy'); 
    xequil = fsolve(fun,xequil,options); 
    toc(t2);
    save('xequil.mat')
    
end
toc(t1);
end