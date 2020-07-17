function [x] = myfsolve(f,x0,rtol,steptol,max_iter,h,verbose,method)
f0 = f(x0);
step_size = inf;
relerr = inf;
n=0;
x = x0;
lambda = .1;
decay = 2;
normf = norm(f0);
if strcmp(method,'homotopy')
    
    fun = @(t,z) -finDifJac(f,z,h,1e-3,1e-4)\f(x0);
    [t,y] = ode23(fun,[0,1],x0);
    x = y(end,:);
    if verbose
        fprintf('norm f: %.3e \n',norm(f(x)));
    end

else
    while (step_size>steptol && relerr>rtol && n<max_iter)
        n = n+1;
        J = finDifJac(f,x,h,1e-3,1e-4);

        if strcmp(method,'powell')
            newton_step = -J\f0;
            gradient_step = -J'*f0;
            t = -gradient_step'*J'*f0/norm(J*gradient_step)^2;
            s = -2*t*gradient_step'*(newton_step - t*gradient_step)/norm(newton_step - t*gradient_step)^2;
            trust_region = t*norm(gradient_step);
            if norm(newton_step) <= trust_region
                step = newton_step;
                step_kind = 'newton';
            elseif norm(gradient_step) <= trust_region
                step = t*gradient_step + s*(newton_step - t*gradient_step);
                step_kind = 'mixed';
            else
                step = t*gradient_step;
                step_kind = 'gradient';
            end
            x = x + step;
            f1 = f(x);
            relerr = norm(f1-f0);
            normf = norm(f1);
            step_size = norm(step);
            f0 = f1;
        elseif strcmp(method,'newton')
            c = 1e-6;
            tau = .5;
            
            newton_step = -J\f0;
            step_size = norm(newton_step);
            newton_dir = newton_step/step_size;
            m = f0'*J*newton_dir;
            t = -c*m;
            j = 0;
            while (norm(f(x)) - norm(f(x+step_size*newton_dir))) < (step_size*t)
                j = j+1;
                if j>100
                    warning('j>100');
                end
                step_size = step_size*tau;
            end
            step = step_size*newton_dir;
            x = x + step;
            f1 = f(x);
            normf = norm(f1);
            relerr = norm(f1-f0);
            step_size = norm(step);
            f0 = f1;
            step_kind = 'newton';

        elseif strcmp(method,'gradient')
            gradient_step = -J'*f0;
            step = gradient_step;
            x = x + step;
            f1 = f(x);
            normf = norm(f1);
            relerr = norm(f1-f0);
            step_size = norm(step);
            f0 = f1;
            step_kind = 'gradient';
        elseif strcmp(method,'lm')
            J2 = J'*J;
            step1 = (J2 + lambda*diag(diag(J2)))\gradient_step;
            step2 = (J2 + lambda/decay*diag(diag(J2)))\gradient_step;
            f1 = f(x+step1);
            f2 = f(x+step2);
            normf1 = norm(f1);
            normf2 = norm(f2);
            if (normf2>normf) && (normf1>normf)
                lambda = lambda*decay;
                continue
            else
                if (normf2<normf) && (normf1<normf)
                    x = x + step2;
                    normf = normf2;
                    relerr = norm(f2-f0);
                    step_size = norm(step2);
                    f0 = f2;
                    lambda = lambda/decay;
                    step_kind = 'newtonish';
                elseif (normf2<normf) && (normf1>normf)
                    x = x + step2;
                    normf = norm(f2);
                    relerr = norm(f2-f0);
                    step_size = norm(step2);
                    f0 = f2;
                    lambda = lambda/decay;
                    step_kind = 'newtonish';
                elseif (normf2>normf) && (normf1<normf)
                    x = x + step1;
                    normf = norm(f1);
                    relerr = norm(f1-f0);
                    step_size = norm(step1);
                    f0 = f1;
                    step_kind = 'gradientish';
                end
            end
        end

        if verbose
            fprintf("step %d: norm(f): %.3e, step size: %.3e, change f: %.3e, %s \n",n,normf,step_size, relerr,step_kind);
        end
    end
end
end

% y = f^2
% dy/dx = ff'

function [xmin] = cubmin(a,fa,fpa,b,fb,c,fc)
%     Finds the minimizer for a cubic polynomial that goes through the
%     points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa
%     f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    C = fpa;
    db = b - a;
    dc = c - a;
    denom = (db * dc)^2 * (db - dc);
    d1 = zeros(2, 2);
    d1(0, 0) = dc^2;
    d1(0, 1) = -db^2;
    d1(1, 0) = -dc^3;
    d1(1, 1) = db^3;
    AB = d1*[fb - fa - C * db; fc - fa - C * dc];
    A = AB(1)/denom;
    B = AB(2)/denom;
    radical = B * B - 3 * A * C;
    xmin = a + (-B + sqrt(radical)) / (3 * A);
end

function [xmin] = quadmin(a,fa,fpa,b,fb)
%     Finds the minimizer for a quadratic polynomial that goes through
%     the points (a,fa), (b,fb) with derivative at a of fpa.
%     f(x) = B*(x-a)^2 + C*(x-a) + D
    D = fa;
    C = fpa;
    db = b - a;
    B = (fb - D - C * db) / (db * db);
    xmin = a - C / (2.0 * B);
end
