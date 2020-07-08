%% apply boundary conditions and return Fourier-Zernike coefficients for R & Z

function [aR,aZ] = bc(x,bndryR,bndryZ,NFP,M,N,iM,symm)

% constants
dimZern = (M+1)^2;
MM = ceil(3*M/2);
NN = ceil(3*N/2);
dimFourM  = 2*M+1;
dimFourMM = 2*MM+1;
dimFourN  = 2*N+1;
dimFourNN = 2*NN+1;
Mpad = (dimFourMM-dimFourM)/2;
Npad = (dimFourNN-dimFourN)/2;

% Fourier-Zernike coefficients
if symm
    ssf = [repmat([false(M,1);true(M+1,1);iM(1:dimZern-dimFourM)<0;iM(1:dimZern-dimFourM)>=0],[N,1]);repmat([true(M,1);false(M+1,1);iM(1:dimZern-dimFourM)>=0;iM(1:dimZern-dimFourM)<0],[N+1,1])];
    X = zeros((2*dimZern-dimFourM)*dimFourN,1);  X(ssf) = x;  X = reshape(X,[2*dimZern-dimFourM,dimFourN]);
else
    ssi = logical([ones(2*M,1); 0; ones(2*(dimZern-dimFourM),1)]);
    ssf = [true((2*dimZern-dimFourM)*(dimFourN-1),1); ssi];
    y = zeros((2*dimZern-dimFourM)*dimFourN,1);  y(ssf) = x;  X = reshape(y,[],dimFourN);
end
X(dimFourM,dimFourN) = -sum(sum(X(M+1:dimFourM,N+1:dimFourN)));  % lambda(v=0,z=0) = 0
aL = X(1:dimFourM,:);
aR = zeros(dimZern,dimFourN);  aR(1:dimZern-dimFourM,:) = X(dimFourM+1:dimZern,:);
aZ = zeros(dimZern,dimFourN);  aZ(1:dimZern-dimFourM,:) = X(dimZern+1:end,:);

% theta & zeta
L = four2phys(four2phys([zeros(Mpad,dimFourNN);[zeros(dimFourM,Npad),aL,zeros(dimFourM,Npad)];zeros(Mpad,dimFourNN)]')');
dv = 2*pi/dimFourMM;        v = (0:dv:(2*pi-dv))';   t = pi - v + L;
dz = 2*pi/(NFP*dimFourNN);  z = 0:dz:(2*pi/NFP-dz);  p = -z;

% format boundary wave numbers
[lm,ln] = size(bndryR);  I = ones(dimFourMM,dimFourNN,(lm+1)/2,ln);
t   = I.*t;
p   = I.*p;
m   = I.*permute(0:(lm-1)/2,[4,3,2,1]);
n   = I.*permute(-(ln-1)/2:(ln-1)/2,[4,3,1,2])*NFP;
sRb = I.*permute(bndryR((lm+1)/2:-1:1,:),[4,3,1,2]);
sZb = I.*permute(bndryZ((lm+1)/2:-1:1,:),[4,3,1,2]);
cRb = I.*permute(bndryR((lm+1)/2:end,:), [4,3,1,2]);
cZb = I.*permute(bndryZ((lm+1)/2:end,:), [4,3,1,2]);

% boundary surface
Rb = sum(sum(sRb.*sin(m.*t-n.*p),3),4)+sum(sum(cRb.*cos(m.*t-n.*p),3),4);
Zb = sum(sum(sZb.*sin(m.*t-n.*p),3),4)+sum(sum(cZb.*cos(m.*t-n.*p),3),4);

% boundary Fourier modes
aRb = phys2four(phys2four(Rb)')';  aRb = aRb(Mpad+(1:dimFourM),Npad+(1:dimFourN));
aZb = phys2four(phys2four(Zb)')';  aZb = aZb(Mpad+(1:dimFourM),Npad+(1:dimFourN));

% boundary coefficients
for i = 1:dimFourM
    aR(end-dimFourM+i,:) = aRb(end-dimFourM+i,:) - sum(aR(iM==i-M-1,:));
    aZ(end-dimFourM+i,:) = aZb(end-dimFourM+i,:) - sum(aZ(iM==i-M-1,:));
end

end
