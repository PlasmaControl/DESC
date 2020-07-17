%% plot force balance error

function [] = plot_f(x,Pres,Iota,Psi,bndryR,bndryZ,NFP,M,N,lm,ln,symm,squr)

bndryR = reshape(bndryR,[lm,ln]);
bndryZ = reshape(bndryZ,[lm,ln]);

% constants
mu0 = 4*pi/1e7;
dimZern = (M+1)^2;
dimFour = 2*N+1;

% sample nodes
dz = 2*pi/6;  z = 0:dz:(2*pi-dz);
[vth,rho] = meshgrid(0:pi/64:2*pi,0:1e-2:1);
r = rho(:);  v = vth(:);  dim = length(r);
axn = (r == 0);
mdn = find(r == 0.5,1);

% interpolation matrices
[iM,ZERN,ZERNr,ZERNv,ZERNrr,ZERNvv,ZERNrv,ZERNrrv,ZERNrvv,ZERNrrvv] = zernfun(M,r,v);

% profiles
cP = zeros(dimZern,1);  cP(find(iM==0,length(Pres))) = Pres;
cI = zeros(dimZern,1);  cI(find(iM==0,length(Iota))) = Iota;
presr = ZERNr*cP;
iota  = ZERN *cI;
iotar = ZERNr*cI;
psir  = 2*Psi*r;
psirr = 2*Psi;

% state variables
[aR,aZ] = bc(x,bndryR,bndryZ,NFP,M,N,iM,symm,squr);

% toroidal derivatives
dk = (-(dimFour-1)/2:(dimFour-1)/2);
aRz  = dk.*fliplr(aR) *NFP;
aZz  = dk.*fliplr(aZ) *NFP;
aRzz = dk.*fliplr(aRz)*NFP;
aZzz = dk.*fliplr(aZz)*NFP;

% Zernike coefficients
cR   = four2phys(aR')';
cZ   = four2phys(aZ')';
cRz  = four2phys(aRz')';
cZz  = four2phys(aZz')';
cRzz = four2phys(aRzz')';
cZzz = four2phys(aZzz')';

% interpolation
cR   = interpfour(cR',z)';      cZ   = interpfour(cZ',z)';
cRz  = interpfour(cRz',z)';     cZz  = interpfour(cZz',z)';
cRzz = interpfour(cRzz',z)';    cZzz = interpfour(cZzz',z)';

% ---- partial derivatives & conversion to physical space --------------- %

R     = ZERN    *cR;            Z     = ZERN    *cZ;
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

% ---- covariant basis vectors ------------------------------------------ %

er(:,:,1)   = Rr;    er(:,:,3)   = Zr;    er(:,:,2)   = zeros(dim,6);
ev(:,:,1)   = Rv;    ev(:,:,3)   = Zv;    ev(:,:,2)   = zeros(dim,6);
ez(:,:,1)   = Rz;    ez(:,:,3)   = Zz;    ez(:,:,2)   = -R;
err(:,:,1)  = Rrr;   err(:,:,3)  = Zrr;   err(:,:,2)  = zeros(dim,6);
erv(:,:,1)  = Rrv;   erv(:,:,3)  = Zrv;   erv(:,:,2)  = zeros(dim,6);
erz(:,:,1)  = Rzr;   erz(:,:,3)  = Zzr;   erz(:,:,2)  = zeros(dim,6);
evv(:,:,1)  = Rvv;   evv(:,:,3)  = Zvv;   evv(:,:,2)  = zeros(dim,6);
evz(:,:,1)  = Rzv;   evz(:,:,3)  = Zzv;   evz(:,:,2)  = zeros(dim,6);
ezr(:,:,1)  = Rzr;   ezr(:,:,3)  = Zzr;   ezr(:,:,2)  = -Rr;
ezv(:,:,1)  = Rzv;   ezv(:,:,3)  = Zzv;   ezv(:,:,2)  = -Rv;
ezz(:,:,1)  = Rzz;   ezz(:,:,3)  = Zzz;   ezz(:,:,2)  = -Rz;
ervv(:,:,1) = Rrvv;  ervv(:,:,3) = Zrvv;  ervv(:,:,2) = zeros(dim,6);
ervz(:,:,1) = Rzrv;  ervz(:,:,3) = Zzrv;  ervz(:,:,2) = zeros(dim,6);
ezrv(:,:,1) = Rzrv;  ezrv(:,:,3) = Zzrv;  ezrv(:,:,2) = -Rrv;

% ---- nonlinear calculations ------------------------------------------- %

% Jacobian
g = dot(er,cross(ev,ez,3),3);

% Jacobian derivatives
gr  = dot(err,cross(ev,ez,3),3) + dot(er,cross(erv,ez,3),3) + dot(er,cross(ev,ezr,3),3);
gv  = dot(erv,cross(ev,ez,3),3) + dot(er,cross(evv,ez,3),3) + dot(er,cross(ev,ezv,3),3);
gz  = dot(erz,cross(ev,ez,3),3) + dot(er,cross(evz,ez,3),3) + dot(er,cross(ev,ezz,3),3);
% rho=0 terms only
grr  = R.*(Rr.*Zrrv - Zr.*Rrrv + 2.*Rrr.*Zrv - 2.*Rrv.*Zrr) + 2.*Rr.*(Zrv.*Rr - Rrv.*Zr);
grv  = R.*(Zrvv.*Rr - Rrvv.*Zr);
gzr  = Rz.*(Rr.*Zrv - Rrv.*Zr) + R.*(Rzr.*Zrv + Rr.*Zzrv - Rzrv.*Zr - Rrv.*Zzr);
grrv = 2.*Rrv.*(Zrv.*Rr - Rrv.*Zr) + 2.*Rr.*(Zrvv.*Rr - Rrvv.*Zr) + R.*(Rr.*Zrrvv - Zr.*Rrrvv + 2.*Rrr.*Zrvv - Rrv.*Zrrv - 2.*Zrr.*Rrvv + Zrv.*Rrrv);

% B contravariant components
BZ = psir ./ (2*pi*g);
BV = iota .* BZ;
BZ(axn,:) = Psi ./ (pi.*gr(axn,:));
BV(axn,:) = Psi.*iota(axn) ./ (pi.*gr(axn,:));

% B^{zeta} derivatives
BZr = psirr ./ (2*pi*g) - (psir.*gr) ./ (2*pi*g.^2);
BZv = - (psir.*gv) ./ (2*pi*g.^2);
BZz = - (psir.*gz) ./ (2*pi*g.^2);
BZr(axn,:) = - (psirr.*grr(axn,:)) ./ (4*pi*gr(axn,:).^2);
BZv(axn,:) = 0;
BZz(axn,:) = - (psirr.*gzr(axn,:)) ./ (2*pi*gr(axn,:).^2);
% rho=0 terms only
BZrv = psirr.*(2*gr.*grr.*grv - gr.^2.*grrv) ./ (4*pi*gr.^4);

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

% force balance error
Fr = (JV.*BZ - JZ.*BV) - presr;
Fv =  JR.*BZ;
Fz = -JR.*BV;

% contravariant basis vectors
eR = cross(ev,ez,3)./g;  eV = cross(ez,er,3)./g;  eZ = cross(er,ev,3)./g;
eR(axn,:,:) = cross(erv(axn,:,:),ez(axn,:,:),3) ./ gr(axn,:);
eV(axn,:,:) = cross(ez(axn,:,:),er(axn,:,:),3);
eZ(axn,:,:) = cross(er(axn,:,:),ev(axn,:,:),3);

% metric coefficients
gRR = dot(eR,eR,3);       gVV = dot(eV,eV,3);       gZZ = dot(eZ,eZ,3);
gRV = dot(eR,eV,3);       gVZ = dot(eV,eZ,3);       gRZ = dot(eR,eZ,3);

% force balance error magnitude
Fmag = sqrt(gRR.*Fr.^2 + gVV.*Fv.^2 + gZZ.*Fz.^2 + 2.*gRV.*Fr.*Fv + 2.*gVZ.*Fv.*Fz + 2.*gRZ.*Fr.*Fz);
Pmag = sqrt(gRR.*presr.^2);

% magnitude & sign of direction vectors
beta = BZ.*eV - BV.*eZ;
radial  = sqrt(gRR) .* sign(dot(eR,er,3));
helical = sqrt(gVV.*BZ.^2 + gZZ.*BV.^2 - 2.*gVZ.*BV.*BZ) .* sign(dot(beta,ev,3)).*sign(dot(beta,ez,3));
helical(axn,:) = sqrt(gVV(axn,:).*BZ(axn,:).^2) .* sign(-BV(axn,:).*BZ(axn,:));

% force balance error
Frho = ((JV.*BZ - JZ.*BV) - presr) .* radial;
Fbta = JR .* helical;

% ---- plotting --------------------------------------------------------- %

zeta = {'0','\pi/3','2\pi/3','\pi','4\pi/3','5\pi/3'};

% Frho grid
figure('units','normalized','outerposition',[0 0 1 1]);
set(gcf,'color','w')
for i = 1:6
    if N > 0;  subplot(2,3,i);  end
    hold on
    Frhoi = reshape(Frho(:,i).*g(:,i),size(vth));
    p=pcolor(vth,rho,Frhoi); p.EdgeColor='none'; colorbar
    xlabel('\vartheta'), ylabel('\rho'), xlim([0,2*pi])
    if i == 1;  title('F_{\rho} ||\nabla\rho|| \surd g [N]')
    else; title(['N_{FP}\zeta = ',zeta{i}]);  end
    set(gca,'FontSize',16,'LineWidth',2), box on, hold off
    if N == 0;  break;  end
end

% Fbeta grid
figure('units','normalized','outerposition',[0 0 1 1]);
set(gcf,'color','w')
for i = 1:6
    if N > 0;  subplot(2,3,i);  end
    hold on
    Fbetai = reshape(Fbta(:,i).*g(:,i),size(vth));
    p=pcolor(vth,rho,Fbetai); p.EdgeColor='none'; colorbar
    xlabel('\vartheta'), ylabel('\rho'), xlim([0,2*pi])
    if i == 1;  title('F_{\beta} ||\beta|| \surd g [N]')
    else; title(['N_{FP}\zeta = ',zeta{i}]);  end
    set(gca,'FontSize',16,'LineWidth',2), box on, hold off
    if N == 0;  break;  end
end

% force error percent space
figure('units','normalized','outerposition',[0 0 1 1]);
set(gcf,'color','w')
for i = 1:6
    if N > 0;  subplot(2,3,i);  end
    hold on
    Fi = reshape(Fmag(:,i)./Pmag(mdn,i),size(vth));
    Ri = reshape(R(:,i),size(vth));
    Zi = reshape(Z(:,i),size(vth));
    levels = [-16 -4 -3 -2 -1];
    contourf(Ri,Zi,log10(Fi),levels)
    c=colorbar; caxis([-4 0]), colormap(parula(4))
    c.YTick = [-3.5 -2.5 -1.5 -0.5];
    c.YTickLabel = {'<0.1%','0.1%-1%','1%-10%','10%-100%'};
    xlabel('R'), ylabel('Z'), axis equal
    if i == 1;  title('||F|| / ||\nabla p(\rho=1/2)||')
    else; title(['N_{FP}\zeta = ',zeta{i}]);  end
    set(gca,'FontSize',16,'LineWidth',2), box on, hold off
    if N == 0;  break;  end
end

end
