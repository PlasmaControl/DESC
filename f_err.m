%% f=f(x) is the system of equations for the equilibrium force balance errors

function f = f_err(x,cP,cI,Psi,bndryR,bndryZ,NFP,M,N,iM,...
    ZERN_C0,ZERNr_C0,ZERNv_C0,ZERNrr_C0,ZERNvv_C0,ZERNrv_C0,ZERNrrv_C0,ZERNrvv_C0,ZERNrrvv_C0,...
    ZERN_S0,ZERNr_S0,ZERNv_S0,ZERNrr_S0,ZERNvv_S0,ZERNrv_S0,ZERNrrv_S0,ZERNrvv_S0,ZERNrrvv_S0,...
    ZERN_C1,ZERNr_C1,ZERNv_C1,ZERNrr_C1,ZERNvv_C1,ZERNrv_C1,ZERNrrv_C1,ZERNrvv_C1,ZERNrrvv_C1,...
    ZERN_S1,ZERNr_S1,ZERNv_S1,ZERNrr_S1,ZERNvv_S1,ZERNrv_S1,ZERNrrv_S1,ZERNrvv_S1,ZERNrrvv_S1,...
    rC0,drC0,dvC0,rS0,drS0,dvS0,rC1,drC1,dvC1,rS1,drS1,dvS1,symm,squr)

%% ---- constants & profiles -------------------------------------------- %

% constants
dimZern = (M+1)^2;
dimFour = 2*N+1;
if squr;  Mnodes = M;            numSlc = dimFour;
else;     Mnodes = ceil(1.5*M);  numSlc = 2*ceil(1.5*N)+1;  end
Npad = (numSlc-dimFour)/2;

%% ---- toroidal Fourier transform -------------------------------------- %

% state variables
[aR,aZ] = bc(x,bndryR,bndryZ,NFP,M,N,iM,symm);

% toroidal derivatives
dk = (-(dimFour-1)/2:(dimFour-1)/2);
aRz  = dk.*fliplr(aR) *NFP;
aZz  = dk.*fliplr(aZ) *NFP;
aRzz = dk.*fliplr(aRz)*NFP;
aZzz = dk.*fliplr(aZz)*NFP;

% toroidal expansion
aR   = [zeros(dimZern,Npad), aR,   zeros(dimZern,Npad)];
aZ   = [zeros(dimZern,Npad), aZ,   zeros(dimZern,Npad)];
aRz  = [zeros(dimZern,Npad), aRz,  zeros(dimZern,Npad)];
aZz  = [zeros(dimZern,Npad), aZz,  zeros(dimZern,Npad)];
aRzz = [zeros(dimZern,Npad), aRzz, zeros(dimZern,Npad)];
aZzz = [zeros(dimZern,Npad), aZzz, zeros(dimZern,Npad)];

% Zernike coefficients
cR   = four2phys(aR')';
cZ   = four2phys(aZ')';
cRz  = four2phys(aRz')';
cZz  = four2phys(aZz')';
cRzz = four2phys(aRzz')';
cZzz = four2phys(aZzz')';

%% ---- force balance errors -------------------------------------------- %

if symm
    % symmetry half domain
    cR(:,((numSlc+1)/2+1):end)   = [];
    cZ(:,((numSlc+1)/2+1):end)   = [];
    cRz(:,((numSlc+1)/2+1):end)  = [];
    cZz(:,((numSlc+1)/2+1):end)  = [];
    cRzz(:,((numSlc+1)/2+1):end) = [];
    cZzz(:,((numSlc+1)/2+1):end) = [];
    % force balance errors
    [frho_C0,~] = fberr(Psi,cR(:,1),cZ(:,1),cRz(:,1),cZz(:,1),cRzz(:,1),cZzz(:,1),cP,cI,ZERN_C0,ZERNr_C0,ZERNv_C0,ZERNrr_C0,ZERNvv_C0,ZERNrv_C0,ZERNrrv_C0,ZERNrvv_C0,ZERNrrvv_C0,rC0,drC0,dvC0,Mnodes);
    [~,fbta_S0] = fberr(Psi,cR(:,1),cZ(:,1),cRz(:,1),cZz(:,1),cRzz(:,1),cZzz(:,1),cP,cI,ZERN_S0,ZERNr_S0,ZERNv_S0,ZERNrr_S0,ZERNvv_S0,ZERNrv_S0,ZERNrrv_S0,ZERNrvv_S0,ZERNrrvv_S0,rS0,drS0,dvS0,Mnodes);
    [frho_C1,~] = fberr(Psi,cR(:,2:end),cZ(:,2:end),cRz(:,2:end),cZz(:,2:end),cRzz(:,2:end),cZzz(:,2:end),cP,cI,ZERN_C1,ZERNr_C1,ZERNv_C1,ZERNrr_C1,ZERNvv_C1,ZERNrv_C1,ZERNrrv_C1,ZERNrvv_C1,ZERNrrvv_C1,rC1,drC1,dvC1,Mnodes);
    [~,fbta_S1] = fberr(Psi,cR(:,2:end),cZ(:,2:end),cRz(:,2:end),cZz(:,2:end),cRzz(:,2:end),cZzz(:,2:end),cP,cI,ZERN_S1,ZERNr_S1,ZERNv_S1,ZERNrr_S1,ZERNvv_S1,ZERNrv_S1,ZERNrrv_S1,ZERNrvv_S1,ZERNrrvv_S1,rS1,drS1,dvS1,Mnodes);
    % output
    f = [frho_C0(:); frho_C1(:); fbta_S0(:); fbta_S1(:)];
else
    % force balance errors
    [frho_C1,~] = fberr(Psi,cR,cZ,cRz,cZz,cRzz,cZzz,cP,cI,ZERN_C1,ZERNr_C1,ZERNv_C1,ZERNrr_C1,ZERNvv_C1,ZERNrv_C1,ZERNrrv_C1,ZERNrvv_C1,ZERNrrvv_C1,rC1,drC1,dvC1,Mnodes);
    [~,fbta_S1] = fberr(Psi,cR,cZ,cRz,cZz,cRzz,cZzz,cP,cI,ZERN_S1,ZERNr_S1,ZERNv_S1,ZERNrr_S1,ZERNvv_S1,ZERNrv_S1,ZERNrrv_S1,ZERNrvv_S1,ZERNrrvv_S1,rS1,drS1,dvS1,Mnodes);
    % output
    f = [frho_C1(:); fbta_S1(:)];
    if squr;  f(1) = [];  end
end

end

function [frho,fbta] = fberr(Psi,cR,cZ,cRz,cZz,cRzz,cZzz,cP,cI,ZERN,ZERNr,ZERNv,ZERNrr,ZERNvv,ZERNrv,ZERNrrv,ZERNrvv,ZERNrrvv,r,dr,dv,M)

% constants
mu0 = 4*pi/1e7;
numSlc = size(cR,2);
numPts = length(r);
dz = pi/numSlc;

% profiles
presr = ZERNr*cP;
iota  = ZERN *cI;
iotar = ZERNr*cI;
psir  = 2*Psi*r;
psirr = 2*Psi;
axn   = (r == 0);

%% ---- partial derivatives & conversion to physical space -------------- %

R     = ZERN    *cR;
Rr    = ZERNr   *cR;            Zr    = ZERNr   *cZ;
Rv    = ZERNv   *cR;            Zv    = ZERNv   *cZ;
Rz    = ZERN    *cRz;           Zz    = ZERN    *cZz;
Rrr   = ZERNrr  *cR;            Zrr   = ZERNrr  *cZ;
Rrv   = ZERNrv  *cR;            Zrv   = ZERNrv  *cZ;
Rvv   = ZERNvv  *cR;            Zvv   = ZERNvv  *cZ;
Rzr   = ZERNr   *cRz;           Zzr   = ZERNr   *cZz;
Rzv   = ZERNv   *cRz;           Zzv   = ZERNv   *cZz;
Rzz   = ZERN    *cRzz;          Zzz   = ZERN    *cZzz;
Rrrv  = ZERNrrv *cR;            Zrrv  = ZERNrrv *cZ;
Rrvv  = ZERNrvv *cR;            Zrvv  = ZERNrvv *cZ;
Rzrv  = ZERNrv  *cRz;           Zzrv  = ZERNrv  *cZz;
Rrrvv = ZERNrrvv*cR;            Zrrvv = ZERNrrvv*cZ;

%% ---- covariant basis vectors ----------------------------------------- %
er = zeros(numPts,numSlc,3);
ev = zeros(numPts,numSlc,3);
ez = zeros(numPts,numSlc,3);
err = zeros(numPts,numSlc,3);
erv = zeros(numPts,numSlc,3);
erz = zeros(numPts,numSlc,3);
evv = zeros(numPts,numSlc,3);
evz = zeros(numPts,numSlc,3);
ezr = zeros(numPts,numSlc,3);
ezv = zeros(numPts,numSlc,3);
ezz = zeros(numPts,numSlc,3);
ervv = zeros(numPts,numSlc,3);
ervz = zeros(numPts,numSlc,3);
ezrv = zeros(numPts,numSlc,3);



er(:,:,1)   = Rr;    er(:,:,3)   = Zr;    er(:,:,2)   = zeros(numPts,numSlc);
ev(:,:,1)   = Rv;    ev(:,:,3)   = Zv;    ev(:,:,2)   = zeros(numPts,numSlc);
ez(:,:,1)   = Rz;    ez(:,:,3)   = Zz;    ez(:,:,2)   = -R;
err(:,:,1)  = Rrr;   err(:,:,3)  = Zrr;   err(:,:,2)  = zeros(numPts,numSlc);
erv(:,:,1)  = Rrv;   erv(:,:,3)  = Zrv;   erv(:,:,2)  = zeros(numPts,numSlc);
erz(:,:,1)  = Rzr;   erz(:,:,3)  = Zzr;   erz(:,:,2)  = zeros(numPts,numSlc);
evv(:,:,1)  = Rvv;   evv(:,:,3)  = Zvv;   evv(:,:,2)  = zeros(numPts,numSlc);
evz(:,:,1)  = Rzv;   evz(:,:,3)  = Zzv;   evz(:,:,2)  = zeros(numPts,numSlc);
ezr(:,:,1)  = Rzr;   ezr(:,:,3)  = Zzr;   ezr(:,:,2)  = -Rr;
ezv(:,:,1)  = Rzv;   ezv(:,:,3)  = Zzv;   ezv(:,:,2)  = -Rv;
ezz(:,:,1)  = Rzz;   ezz(:,:,3)  = Zzz;   ezz(:,:,2)  = -Rz;
ervv(:,:,1) = Rrvv;  ervv(:,:,3) = Zrvv;  ervv(:,:,2) = zeros(numPts,numSlc);
ervz(:,:,1) = Rzrv;  ervz(:,:,3) = Zzrv;  ervz(:,:,2) = zeros(numPts,numSlc);
ezrv(:,:,1) = Rzrv;  ezrv(:,:,3) = Zzrv;  ezrv(:,:,2) = -Rrv;

%% ---- nonlinear calculations ------------------------------------------ %

% Jacobian
g = dot(er,cross(ev,ez,3),3);

% Jacobian derivatives
gr = dot(err,cross(ev,ez,3),3) + dot(er,cross(erv,ez,3),3) + dot(er,cross(ev,ezr,3),3);
gv = dot(erv,cross(ev,ez,3),3) + dot(er,cross(evv,ez,3),3) + dot(er,cross(ev,ezv,3),3);
gz = dot(erz,cross(ev,ez,3),3) + dot(er,cross(evz,ez,3),3) + dot(er,cross(ev,ezz,3),3);
% rho=0 terms only
grr  = R.*(Rr.*Zrrv - Zr.*Rrrv + 2.*Rrr.*Zrv - 2.*Rrv.*Zrr) + 2.*Rr.*(Zrv.*Rr - Rrv.*Zr);
grv  = R.*(Zrvv.*Rr - Rrvv.*Zr);
gzr  = Rz.*(Rr.*Zrv - Rrv.*Zr) + R.*(Rzr.*Zrv + Rr.*Zzrv - Rzrv.*Zr - Rrv.*Zzr);
grrv = 2.*Rrv.*(Zrv.*Rr - Rrv.*Zr) + 2.*Rr.*(Zrvv.*Rr - Rrvv.*Zr) + R.*(Rr.*Zrrvv - Zr.*Rrrvv + 2.*Rrr.*Zrvv - Rrv.*Zrrv - 2.*Zrr.*Rrvv + Zrv.*Rrrv);

% B contravariant components
BZ = psir ./ (2*pi*g);
BV = iota .* BZ;

% B^{zeta} derivatives
BZr = psirr ./ (2*pi*g) - (psir.*gr) ./ (2*pi*g.^2);
BZv = - (psir.*gv) ./ (2*pi*g.^2);
BZz = - (psir.*gz) ./ (2*pi*g.^2);
% rho=0 terms only
BZrv = psirr.*(2*grr.*grv - gr.*grrv) ./ (4*pi*gr.^3);

% magnetic axis
BZ(axn,:)  = Psi ./ (pi.*gr(axn,:));
BV(axn,:)  = Psi.*iota(axn) ./ (pi.*gr(axn,:));
BZr(axn,:) = - (psirr.*grr(axn,:)) ./ (4*pi*gr(axn,:).^2);
BZv(axn,:) = 0;
BZz(axn,:) = - (psirr.*gzr(axn,:)) ./ (2*pi*gr(axn,:).^2);

% covariant B-component derivatives
Bv_r = BZr.*dot(iota.*ev+ez,ev,3) + BZ.*dot(iotar.*ev+iota.*erv+ezr,ev,3) + BZ.*dot(iota.*ev+ez,erv,3);
Bz_r = BZr.*dot(iota.*ev+ez,ez,3) + BZ.*dot(iotar.*ev+iota.*erv+ezr,ez,3) + BZ.*dot(iota.*ev+ez,ezr,3);
Br_v = BZv.*dot(iota.*ev+ez,er,3) + BZ.*dot(iota.*evv+ezv,er,3) + BZ.*dot(iota.*ev+ez,erv,3);
Bz_v = BZv.*dot(iota.*ev+ez,ez,3) + BZ.*dot(iota.*evv+ezv,ez,3) + BZ.*dot(iota.*ev+ez,ezv,3);
Br_z = BZz.*dot(iota.*ev+ez,er,3) + BZ.*dot(iota.*evz+ezz,er,3) + BZ.*dot(iota.*ev+ez,erz,3);
Bv_z = BZz.*dot(iota.*ev+ez,ev,3) + BZ.*dot(iota.*evz+ezz,ev,3) + BZ.*dot(iota.*ev+ez,evz,3);
% rho=0 terms only
Bz_rv = BZrv.*dot(ez,ez,3) + BZ.*dot(iota.*ervv+2*ezrv,ez,3);
Bv_zr = BZz.*dot(ez,erv,3) + BZ.*(dot(ezz,erv,3)+dot(ez,ervz,3));

% contravariant J-components
JR = (Bz_v - Bv_z) ./ mu0;
JV = (Br_z - Bz_r) ./ mu0;
JZ = (Bv_r - Br_v) ./ mu0;
JR(axn,:) = (Bz_rv(axn,:) - Bv_zr(axn,:)) ./ (mu0*gr(axn,:));

% contravariant basis vectors
eR = cross(ev,ez,3)./g;  eV = cross(ez,er,3)./g;  eZ = cross(er,ev,3)./g;
eR(axn,:,:) = cross(erv(axn,:,:),ez(axn,:,:),3) ./ gr(axn,:);
eV(axn,:,:) = cross(ez(axn,:,:),er(axn,:,:),3);
eZ(axn,:,:) = cross(er(axn,:,:),ev(axn,:,:),3);

% metric coefficients
gRR = dot(eR,eR,3);  gVV = dot(eV,eV,3);  gZZ = dot(eZ,eZ,3);  gVZ = dot(eV,eZ,3);

% magnitude & sign of direction vectors
beta = BZ.*eV - BV.*eZ;
radial  = sqrt(gRR) .* sign(dot(eR,er,3));
helical = sqrt(gVV.*BZ.^2 + gZZ.*BV.^2 - 2.*gVZ.*BV.*BZ) .* sign(dot(beta,ev,3)).*sign(dot(beta,ez,3));
helical(axn,:) = sqrt(gVV(axn,:).*BZ(axn,:).^2) .* sign(BZ(axn,:));

% force balance error
Frho = ((JV.*BZ - JZ.*BV) - presr) .* radial;
Fbta = JR .* helical;
if sum(axn) == 0
    vol = g.*dr.*dv.*dz;
    frho = Frho .* vol;
    fbta = Fbta .* vol;
else
    vol0 = g(2*M+2,:)/2.*dr(1).*dv(1).*dz;
    frho0 = mean(Frho(1:2*M+1,:),1) .* vol0;
    fbta0 = mean(Fbta(1:2*M+1,:),1) .* vol0;
    vol  = g(2*M+1:end,:).*dr.*dv.*dz;
    frho = Frho(2*M+1:end,:) .* vol;
    fbta = Fbta(2*M+1:end,:) .* vol;
    frho(1,:) = frho0;
    fbta(1,:) = fbta0;
end

end
