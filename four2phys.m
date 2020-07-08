%% transform from Fourier space to physical space
% c(n) = vector of Fourier coefficients for the modes -N,...,-1,0,1,...,N
% x(t) = sum_{n=-1}^{-N} c(n)*sin(|n|*t)  for n <  0
%      + sum_{n=0}^{N}   c(n)*cos(|n|*t)  for n >= 0
% transform is performed along the first dimension of c (columns)

function x = four2phys(c)

[L,K] = size(c);
N = (L-1)/2;

% pad with negative wavenumbers
a = c(N+1:end,:);              b = c(N:-1:1,:);
a = [a(1,:);     a(2:end,:)/2; a(end:-1:2,:)/2];
b = [zeros(1,K);-b(1:end,:)/2; b(end:-1:1,:)/2];

% inverse Fourier transform
a = a*L;
b = b*L;
c = complex(a,b);
x = real(ifft(c,L,1));

end
