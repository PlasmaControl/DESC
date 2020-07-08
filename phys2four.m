%% transform from physical space to Fourier space
% c(n) = vector of Fourier coefficients for the modes -N,...,-1,0,1,...,N
% x(t) = sum_{n=-1}^{-N} c(n)*sin(|n|*t)  for n <  0
%      + sum_{n=0}^{N}   c(n)*cos(|n|*t)  for n >= 0
% transform is performed along the first dimension of x (columns)

function c = phys2four(x)

L = size(x,1);
N = (L-1)/2;

% Fourier transform
c = fft(x,L,1);
a = real(c)/L;
b = imag(c)/L;

% only positive wavenumbers
a = [a(1,:); 2*a(2:N+1,:)];
b = -2*b(2:N+1,:);
c = [b(end:-1:1,:); a];

end
