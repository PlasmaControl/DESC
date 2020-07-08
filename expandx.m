%% increase spectral resolution
% x1 = state vector at resolution M1, N1
% x2 = state vector at resolution M2, N2

function x2 = expandx(x1, M1, N1, M2, N2, symm)

J1 = (M1+1)^2;
J2 = (M2+1)^2;
B1 = 2*M1+1;
B2 = 2*M2+1;
K1 = 2*N1+1;
K2 = 2*N2+1;
bb = (B2-B1)/2;
kk = (K2-K1)/2;

% Zernike functions modes
Mv2 = zeros(J2,1);  j = 0;
for m = 0:M2
    Mv2(j+1:j+2*m+1) = -m:1:m;
    j = j+2*m+1;
end
Mv1 = Mv2(1:J1);

if symm
    ssf = [repmat([false(M1,1);true(M1+1,1);Mv1(1:J1-B1)<0;Mv1(1:J1-B1)>=0],[N1,1]);repmat([true(M1,1);false(M1+1,1);Mv1(1:J1-B1)>=0;Mv1(1:J1-B1)<0],[N1+1,1])];
    X1 = zeros((2*J1-B1)*K1,1);  X1(ssf) = x1;  X1 = reshape(X1,[2*J1-B1,K1]);
else
    ssi = logical([ones(2*M1,1); 0; ones(2*(J1-B1),1)]);
    ssf = [true((2*J1-B1)*(K1-1),1); ssi];
    y = zeros((2*J1-B1)*K1,1);  y(ssf) = x1;  X1 = reshape(y,[],K1);
end

X2 = zeros(2*J2-B2,K2);
X2(bb+(1:B1),kk+(1:K1)) = X1(1:B1,:);
X2(B2+1:J2-((J2-B2)-(J1-B1)),kk+(1:K1)) = X1(B1+1:J1,:);
X2(J2+1:end-((J2-B2)-(J1-B1)),kk+(1:K1)) = X1(J1+1:end,:);

if symm
    ssf = [repmat([false(M2,1);true(M2+1,1);Mv2(1:J2-B2)<0;Mv2(1:J2-B2)>=0],[N2,1]);repmat([true(M2,1);false(M2+1,1);Mv2(1:J2-B2)>=0;Mv2(1:J2-B2)<0],[N2+1,1])];
    x2 = X2(:);  x2(~ssf) = [];
else
    ssi = logical([ones(2*M2,1); 0; ones(2*(J2-B2),1)]);
    ssf = [true((2*J2-B2)*(K2-1),1); ssi];
    x2 = X2(:);  x2(~ssf) = [];
end

end
