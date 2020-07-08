%% plot flux surfaces

function [] = plot_x(x,bndryR,bndryZ,NFP,M,N,symm)

dz = 2*pi/6;  z = 0:dz:(2*pi-dz);
zeta = {'0','\pi/3','2\pi/3','\pi','4\pi/3','5\pi/3'};

[RHOv,THETAv] = meshgrid(0:1/8:1,0:pi/128:2*pi);
[iM,ZERNv] = zernfun(M,RHOv(:),THETAv(:));
[RHOr,THETAr] = meshgrid(0:1e-2:1,0:pi/6:2*pi);
[~,ZERNr] = zernfun(M,RHOr(:),THETAr(:));

[aR,aZ] = bc(x,bndryR,bndryZ,NFP,M,N,iM,symm);
cR  = four2phys(aR')';     cZ  = four2phys(aZ')';
cR  = interpfour(cR',z)';  cZ  = interpfour(cZ',z)';
R_v = ZERNv*cR;            Z_v = ZERNv*cZ;
R_r = ZERNr*cR;            Z_r = ZERNr*cZ;

figure('units','normalized','outerposition',[0 0 1 1]);
set(gcf,'color','w')

for i = 1:6

    if N > 0;  subplot(2,3,i);  end
    hold on
    
    R = reshape(R_v(:,i),size(RHOv));
    Z = reshape(Z_v(:,i),size(RHOv));
    plot(R(:,1),Z(:,1),'ko-','MarkerFaceColor','k','MarkerSize',6,'LineWidth',2)
    plot(R,Z,'k-','LineWidth',2)
    R = reshape(R_r(:,i),size(RHOr));
    Z = reshape(Z_r(:,i),size(RHOr));
    plot(R',Z','k:','LineWidth',2)
    
    xlabel('R [m]'), ylabel('Z [m]'), title(['N_{FP}\zeta = ',zeta{i}]), axis equal
    set(gca,'FontSize',16,'LineWidth',2), box on, hold off
    
    if N == 0;  break;  end
    
end

end
