%% create initial guess for equilibrium flux surfaces
% boundary surface is scaled proportional to rho

function x = x_init(bndryR,bndryZ,NFP,M,N,lm,ln,symm)

bndryR = reshape(bndryR,[lm,ln]);
bndryZ = reshape(bndryZ,[lm,ln]);

dimZern  = (M+1)^2;
dimFourM = 2*M+1;
dimFourN = 2*N+1;

dz = 2*pi/(NFP*dimFourN);  z = 0:dz:(2*pi/NFP-dz);
[r,v] = meshgrid(0:1e-2:1,0:pi/64:2*pi);
r = r(:);  v = v(:);  dim = length(r);
t = pi - v;  p = -z;

% Zernike polynomial basis
[iM,ZERN] = zernfun(M,r,v);  ZERN(:,dimZern+1:end) = [];

% format boundary wave numbers
[lm,ln] = size(bndryR);  I = ones(dim,dimFourN,(lm+1)/2,ln);
t   = I.*t;
p   = I.*p;
m   = I.*permute(0:(lm-1)/2,[4,3,2,1]);
n   = I.*permute(-(ln-1)/2:(ln-1)/2,[4,3,1,2]);
sRb = I.*permute(bndryR((lm+1)/2:-1:1,:),[4,3,1,2]);
sZb = I.*permute(bndryZ((lm+1)/2:-1:1,:),[4,3,1,2]);
cRb = I.*permute(bndryR((lm+1)/2:end,:),[4,3,1,2]);
cZb = I.*permute(bndryZ((lm+1)/2:end,:),[4,3,1,2]);

% boundary surface
R = sum(sum(sRb.*sin(m.*t-n.*NFP.*p),3),4)+sum(sum(cRb.*cos(m.*t-n.*NFP.*p),3),4);
Z = sum(sum(sZb.*sin(m.*t-n.*NFP.*p),3),4)+sum(sum(cZb.*cos(m.*t-n.*NFP.*p),3),4);
R0 = sum(sum(sRb(1,:,1,:).*sin(-n(1,:,1,:).*NFP.*p(1,:,1,:)),3),4) ...
   + sum(sum(cRb(1,:,1,:).*cos(-n(1,:,1,:).*NFP.*p(1,:,1,:)),3),4);
Z0 = sum(sum(sZb(1,:,1,:).*sin(-n(1,:,1,:).*NFP.*p(1,:,1,:)),3),4) ...
   + sum(sum(cZb(1,:,1,:).*cos(-n(1,:,1,:).*NFP.*p(1,:,1,:)),3),4);

% scale boundary proportional to rho
R = r.*(R-R0) + R0;
Z = r.*(Z-Z0) + Z0;

y = zeros(2*dimZern+dimFourM,dimFourN);

for k = 1:dimFourN
    
    % Zernike coefficients
    cR = ZERN \ R(:,k);
    cZ = ZERN \ Z(:,k);
    
    % Fourier coefficients
    cL = zeros(dimFourM,1);
    
    y(:,k) = [cL; cR; cZ];
    
end

X = phys2four(y')';

% stellarator symmetry indices
if symm
    ssi = [repmat([false(M,1);true(M+1,1);iM<0;iM>=0],[N,1]);repmat([true(M,1);false(M+1,1);iM>=0;iM<0],[N+1,1])];
    x = X(:);  x(~ssi) = [];
else
    ssi = logical([ones(2*M,1); 0; ones(2*dimZern,1)]);
    ssi = [true((2*dimZern+dimFourM)*(dimFourN-1),1); ssi];
    x = X(:);  x(~ssi) = [];
end

end
