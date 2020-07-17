%% compute interpolation (Vandermonde) matrices for Zernike basis
% if c is the vector of Zernike coefficients, then:
% f = ZERN*c gives the values of the function interpolated at the points (r,v)
% df/drho = ZERNr*c, df/dvartheta = ZERNv*c, etc.
% iM = poloidal modes of the Zernike basis functions
% fringe indexing is assumed

function [iM,ZERN,ZERNr,ZERNv,ZERNrr,ZERNvv,ZERNrv,ZERNrrv,ZERNrvv,ZERNrrvv] = zernfun(M,r,v)

dimZern = (M+1)^2;
numPts = length(r);

ZERN     = zeros(numPts,dimZern);
ZERNr    = zeros(numPts,dimZern);
ZERNv    = zeros(numPts,dimZern);
ZERNrr   = zeros(numPts,dimZern);
ZERNvv   = zeros(numPts,dimZern);
ZERNrv   = zeros(numPts,dimZern);
ZERNrrv  = zeros(numPts,dimZern);
ZERNrvv  = zeros(numPts,dimZern);
ZERNrrvv = zeros(numPts,dimZern);

% Zernike functions modes
iL = zeros(dimZern,1);  iM = zeros(dimZern,1);  j = 0;
for m = 0:M
    iL(j+1:j+2*m+1) = [m:1:2*m,2*m-1:-1:m];
    iM(j+1:j+2*m+1) = -m:1:m;
    j = j+2*m+1;
end
% iNorm = zeros(dimZern,1);
% for i = 1:dimZern
%     if iM(i) == 0
%         iNorm(i) = sqrt((iL(i)+1)/pi);
%     else
%         iNorm(i) = sqrt(2*(iL(i)+1)/pi);
%     end
% end

% Jacobi polynomials
J   = zeros(numPts,dimZern);
Jr  = zeros(numPts,dimZern);
Jrr = zeros(numPts,dimZern);
for i = 1:dimZern
    c0 = zeros(iL(i)+1,1); c1 = zeros(iL(i)+1,1); c2 = zeros(iL(i)+1,1);
    for j = 0:(iL(i)-iM(i))/2
        if ((iL(i)+iM(i))/2-j >= 0 && (iL(i)-iM(i))/2-j >= 0)
            c0(2*j+1) = (-1)^j*factorial(iL(i)-j)/(factorial(j)*factorial((iL(i)+iM(i))/2-j)*factorial((iL(i)-iM(i))/2-j));
        end
    end
    for j = 0:iL(i)-1
        c1(j+2) = c0(j+1)*(iL(i)-j);
    end
    for j = 0:iL(i)-2
        c2(j+3) = c1(j+2)*(iL(i)-1-j);
    end
    % Horner's method
    for j = 0:iL(i)
        J(:,i)   = J(:,i).*r   + c0(j+1);
        Jr(:,i)  = Jr(:,i).*r  + c1(j+1);
        Jrr(:,i) = Jrr(:,i).*r + c2(j+1);
    end
end

% trigonometric functions
pos = iM >= 0;
neg = iM < 0;

ZERN(:,pos)   = J(:,pos).*  cos(v.*abs(iM(pos))');
ZERN(:,neg)   = J(:,neg).*  sin(v.*abs(iM(neg))');
ZERNr(:,pos)  = Jr(:,pos).* cos(v.*abs(iM(pos))');
ZERNr(:,neg)  = Jr(:,neg).* sin(v.*abs(iM(neg))');
ZERNrr(:,pos) = Jrr(:,pos).*cos(v.*abs(iM(pos))');
ZERNrr(:,neg) = Jrr(:,neg).*sin(v.*abs(iM(neg))');

ZERNv(:,pos)   = J(:,pos).*  -abs(iM(pos))'.*sin(v.*abs(iM(pos))');
ZERNv(:,neg)   = J(:,neg).*   abs(iM(neg))'.*cos(v.*abs(iM(neg))');
ZERNrv(:,pos)  = Jr(:,pos).* -abs(iM(pos))'.*sin(v.*abs(iM(pos))');
ZERNrv(:,neg)  = Jr(:,neg).*  abs(iM(neg))'.*cos(v.*abs(iM(neg))');
ZERNrrv(:,pos) = Jrr(:,pos).*-abs(iM(pos))'.*sin(v.*abs(iM(pos))');
ZERNrrv(:,neg) = Jrr(:,neg).* abs(iM(neg))'.*cos(v.*abs(iM(neg))');

ZERNvv(:,pos)   = J(:,pos).*  -abs(iM(pos)).^2'.*cos(v.*abs(iM(pos))');
ZERNvv(:,neg)   = J(:,neg).*  -abs(iM(neg)).^2'.*sin(v.*abs(iM(neg))');
ZERNrvv(:,pos)  = Jr(:,pos).* -abs(iM(pos)).^2'.*cos(v.*abs(iM(pos))');
ZERNrvv(:,neg)  = Jr(:,neg).* -abs(iM(neg)).^2'.*sin(v.*abs(iM(neg))');
ZERNrrvv(:,pos) = Jrr(:,pos).*-abs(iM(pos)).^2'.*cos(v.*abs(iM(pos))');
ZERNrrvv(:,neg) = Jrr(:,neg).*-abs(iM(neg)).^2'.*sin(v.*abs(iM(neg))');

end
