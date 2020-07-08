%% interpolate a periodic function in real space using FFT
% y = x(t)

function y = interpfour(x, t)

t = t(:);
a = phys2four(x);
N = (size(a,1)-1)/2;
s = flipud(a(1:N+1,:));
c = a(N+1:end,:);

y = zeros(length(t),size(x,2));
for m = 0:N
    y = y + c(m+1,:).*cos(m*t) + s(m+1,:).*sin(m*t);
end

end
