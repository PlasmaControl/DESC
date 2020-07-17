%% apply boundary conditions and return Fourier-Zernike coefficients for R & Z

function [aR,aZ,aRb_err,aZb_err] = bc(x,bndryR,bndryZ,NFP,M,N,iM,symm,squr)

% constants
if squr;  MM = M;            NN = M;
else;     MM = ceil(1.5*M);  NN = ceil(1.5*N);  end
dimZern = (M+1)^2;
dimFourM = 2*M+1;
dimFourN = 2*N+1;
dimFourMM = 2*MM+1;
dimFourNN = 2*NN+1;
Mpad = (dimFourMM-dimFourM)/2;
Npad = (dimFourNN-dimFourN)/2;

% stellarator symmetry indices
if symm
    ssi = [repmat([false(M,1);true(M+1,1);iM<0;iM>=0],[N,1]);repmat([true(M,1);false(M+1,1);iM>=0;iM<0],[N+1,1])];
    X = zeros((2*dimZern+dimFourM)*dimFourN,1);  X(ssi) = x;  X = reshape(X,[2*dimZern+dimFourM,dimFourN]);
else
    ssi = logical([ones(2*M,1); 0; ones(2*dimZern,1)]);
    ssi = [true((2*dimZern+dimFourM)*(dimFourN-1),1); ssi];
    y = zeros((2*dimZern+dimFourM)*dimFourN,1);  y(ssi) = x;  X = reshape(y,[],dimFourN);
end

% Fourier-Zernike coefficients
aL = X(1:dimFourM,:);
aR = X(dimFourM+1:dimFourM+dimZern,:);
aZ = X(dimFourM+dimZern+1:end,:);

% lambda(v=0,z=0) = 0
aL(end,end) = -sum(sum(aL(M+1:end,N+1:end)));

% theta & phi
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
aRb = phys2four(phys2four(Rb)')';
aZb = phys2four(phys2four(Zb)')';

% toroidal reduction
aRb = aRb(Mpad+(1:dimFourM),Npad+(1:dimFourN));
aZb = aZb(Mpad+(1:dimFourM),Npad+(1:dimFourN));

% boundary errors
aRb_err = zeros(dimFourM,dimFourN);
aZb_err = zeros(dimFourM,dimFourN);
for i = 1:dimFourM
    aRb_err(i,:) = aRb(i,:) - sum(aR(iM==i-M-1,:));
    aZb_err(i,:) = aZb(i,:) - sum(aZ(iM==i-M-1,:));
end

end
