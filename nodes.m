%% calculate collocation nodes
% rC0 = rho coordinates with coseine symmetry in the zeta=0 plane
% vS1 = vartheta coordinates with sine symmetry in the zeta!=0 planes
% dr and dv are the volume elemens surrounding that point

% the radial coordinates (r and dr variables) can be changed
% the poloidal coordinates (v and dv variables) should not be changed

function [rC0,vC0,rS0,vS0,rC1,vC1,rS1,vS1,drC0,dvC0,drS0,dvS0,drC1,dvC1,drS1,dvS1] = nodes(M)

rC0 = zeros(M*(M+1)/2,1);  vC0 = zeros(M*(M+1)/2,1);
rS0 = zeros(M*(M+1)/2,1);  vS0 = zeros(M*(M+1)/2,1);
rC1 = zeros(M^2,1);        vC1 = zeros(M^2,1);
rS1 = zeros((M+1)^2,1);    vS1 = zeros((M+1)^2,1);

drC0 = zeros(M*(M+1)/2,1); dvC0 = zeros(M*(M+1)/2,1);
drS0 = zeros(M*(M+1)/2,1); dvS0 = zeros(M*(M+1)/2,1);
drC1 = zeros(M^2,1);       dvC1 = zeros(M^2,1);
drS1 = zeros((M+1)^2,1);   dvS1 = zeros((M+1)^2,1);

% zeta = 0 plane
i = 1;
for m = 0:M-1
    for j = 0:m
        rC0(i) = (cos((M-m)*pi/M)+1)/2;
        vC0(i) = pi/m*j;
        rS0(i) = (cos((M-m-1)*pi/M)+1)/2;
        vS0(i) = pi/(2*(m+1))*(2*j+1);
        if m == 0
            vC0(i) = 0;
            drC0(i) = (cos((M-m-1)*pi/M)+1)/4;
            drS0(i) = (cos((M-m-1)*pi/M)+cos((M-m-2)*pi/M)+2)/4;
        elseif m == M-1
            drC0(i) = 1-(cos((M-m)*pi/M)+cos((M-m+1)*pi/M)+2)/4;
            drS0(i) = 1-(cos((M-m-1)*pi/M)+cos((M-m)*pi/M)+2)/4;
        else
            drC0(i) = (cos((M-m-1)*pi/M)-cos((M-m+1)*pi/M))/4;
            drS0(i) = (cos((M-m-2)*pi/M)-cos((M-m+0)*pi/M))/4;
        end
        dvC0(i) = 2*pi/(m+1);
        dvS0(i) = 2*pi/(m+1);
        i = i+1;
    end
end

% zeta != 0 planes with cosine symmetry
i = 1;
for m = 0:M-1
    for j = 0:2*m
        rC1(i) = (cos((M-m)*pi/M)+1)/2;
        vC1(i) = 2*pi/(2*m+1)*j;
        if m == 0
            drC1(i) = (cos((M-m-1)*pi/M)+1)/4;
        elseif m == M-1
            drC1(i) = 1-(cos((M-m)*pi/M)+cos((M-m+1)*pi/M)+2)/4;
        else
            drC1(i) = (cos((M-m-1)*pi/M)-cos((M-m+1)*pi/M))/4;
        end
        dvC1(i) = 2*pi/(2*m+1);
        i = i+1;
    end
end

% zeta != 0 planes with sine symmetry
i = 1;
for m = 0:M
    for j = 0:2*m
        rS1(i) = (cos((M-m)*pi/M)+1)/2;
        vS1(i) = 2*pi/(2*m+1)*j;
        if m == 0
            drS1(i) = (cos((M-m-1)*pi/M)+1)/4;
        elseif m == M
            drS1(i) = 1-(cos((M-m)*pi/M)+cos((M-m+1)*pi/M)+2)/4;
        else
            drS1(i) = (cos((M-m-1)*pi/M)-cos((M-m+1)*pi/M))/4;
        end
        dvS1(i) = 2*pi/(2*m+1);
        i = i+1;
    end
end

end
