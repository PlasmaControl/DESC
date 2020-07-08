%% differentiate a periodic function in real space using FFT
% y is the mth derivative of x calculated along dimension dim
% x is defined on a uniform grid from [0,2*pi) 

function y = fourdifft(x,m,dim)

K = size(x,dim);

% wavenumbers
K1 = floor((K-1)/2);          
K2 = (-K/2)*rem(m+1,2)*ones(rem(K+1,2));
if dim == 2
    wave = repmat([(0:K1),K2,(-K1:-1)],[size(x,1),1,size(x,3)]);
elseif dim == 3
    wave = repmat(permute([(0:K1),K2,(-K1:-1)],[1,3,2]),[size(x,1),size(x,2),1]);
else
    error('invalid dimension')
end

% transform, take derivative, inverse transform
y = real(ifft(((1j*wave).^m).*fft(x,[],dim),[],dim));

end
