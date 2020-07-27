%% increase spectral resolution
% x1 = state vector at resolution M1, N1
% x2 = state vector at resolution M2, N2

function x2 = expandx(x1,M1,N1,M2,N2,symm)

dimZern1 = (M1+1)^2;
dimZern2 = (M2+1)^2;
dimFourM1 = 2*M1+1;
dimFourM2 = 2*M2+1;
dimFourN1 = 2*N1+1;
dimFourN2 = 2*N2+1;
mm = (dimFourM2-dimFourM1)/2;
nn = (dimFourN2-dimFourN1)/2;

% Zernike functions modes
iM2 = zeros(dimZern2,1);  j = 0;
for m = 0:M2
    iM2(j+1:j+2*m+1) = -m:1:m;
    j = j+2*m+1;
end
iM1 = iM2(1:dimZern1);

% stellarator symmetry indices
if symm
    ssi = [repmat([false(M1,1);true(M1+1,1);iM1<0;iM1>=0],[N1,1]);repmat([true(M1,1);false(M1+1,1);iM1>=0;iM1<0],[N1+1,1])];
    X1 = zeros((2*dimZern1+dimFourM1)*dimFourN1,1);  X1(ssi) = x1;  X1 = reshape(X1,[2*dimZern1+dimFourM1,dimFourN1]);
else
    ssi = logical([ones(2*M1,1); 0; ones(2*dimZern1,1)]);
    ssi = [true((2*dimZern1+dimFourM1)*(dimFourN1-1),1); ssi];
    y = zeros((2*dimZern1+dimFourM1)*dimFourN1,1);  y(ssi) = x1;  X1 = reshape(y,[],dimFourN1);
end

% expansion
X2 = zeros(2*dimZern2+dimFourM2,dimFourN2);
X2(mm+(1:dimFourM1),nn+(1:dimFourN1)) = X1(1:dimFourM1,:);
X2(dimFourM2+1:dimFourM2+dimZern1,nn+(1:dimFourN1)) = X1(dimFourM1+1:dimFourM1+dimZern1,:);
X2(dimFourM2+dimZern2+1:dimFourM2+dimZern2+dimZern1,nn+(1:dimFourN1)) = X1(dimFourM1+dimZern1+1:end,:);

% stellarator symmetry indices
if symm
    ssi = [repmat([false(M2,1);true(M2+1,1);iM2<0;iM2>=0],[N2,1]);repmat([true(M2,1);false(M2+1,1);iM2>=0;iM2<0],[N2+1,1])];
    x2 = X2(:);  x2(~ssi) = [];
else
    ssi = logical([ones(2*M2,1); 0; ones(2*dimZern2,1)]);
    ssi = [true((2*dimZern2+dimFourM2)*(dimFourN2-1),1); ssi];
    x2 = X2(:);  x2(~ssi) = [];
end

end
