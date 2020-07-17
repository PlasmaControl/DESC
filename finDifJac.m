function [J] = finDifJac(f,x,h0,relerrtol,abserrtol)
    %computes a finite difference approximation to the jacobian of a function f at the point x
    %uses centered difference
    n = length(x);
    m = length(f(x));
    sz = [m,n];
    J = zeros(sz);
    mask = true(sz);
    h = h0*ones(n,1);
    relerr = inf;
    abserr = inf;
    while sum(mask(:))>0
        Jold = J;
        for i=1:n
            xp1 = x+[zeros(i-1,1);h(i);zeros(n-i,1)];
            xm1 = x-[zeros(i-1,1);h(i);zeros(n-i,1)];
            xp2 = x+[zeros(i-1,1);2*h(i);zeros(n-i,1)];
            xm2 = x+[zeros(i-1,1);2*h(i);zeros(n-i,1)];
            fp1 = f(xp1);
            fm1 = f(xm1);
            fp2 = f(xp2);
            fm2 = f(xm2);
            foo = (1/12*fm2 - 2/3*fm1 + 2/3*fp1 - 1/12*fp2)/(h(i));
            J(mask(:,i),i) = foo(mask(:,i));
        end
        abserrmat = abs(J-Jold);
        relerrmat = abserrmat./(abs(Jold) + abs(J));
        relerr = max(relerrmat(:));
        abserr = max(abserrmat(:));
        toohigh = (relerrmat>relerrtol) .* (abserrmat>abserrtol);
        mask(~toohigh) = false;
        colidx = any(toohigh,2);
        colidx = unique(colidx);
        h(colidx) = h(colidx)/1.5;
        if any(h<100*eps)
            warning('step size is close to machine precision! h = %e, max error on last iteration = %e',min(h),max(errmat(:)));
        end
    end
end