%% calculates equilibrium solution

clearvars

% input_Heliotron
input_Dshape

% BC weight on tau equations
w = 1e4;

for k = 1:length(M)
    
    fprintf('\nM = %u\nN = %u\n',M(k),N(k))
    if k == 1  % scaled boundary initial guess
        xequil = x_init(bndryR,bndryZ,NFP,M(k),N(k),lm,ln,symm);
    else       % continue from lower resolution solution
        load('xequil.mat','xequil');
        xequil = expandx(xequil,M(k-1),N(k-1),M(k),N(k),symm);
    end
    
    if squr  % square system
        MM = M(k);
    else     % oversample by 3/2
        MM = ceil(1.5*M(k));
    end
    % collocation nodes
    [rC0,vC0,rS0,vS0,rC1,vC1,rS1,vS1,drC0,dvC0,drS0,dvS0,drC1,dvC1,drS1,dvS1] = nodes(MM);
    % pre-compute interpolation matrices
    [iM,ZERN_C0,ZERNr_C0,ZERNv_C0,ZERNrr_C0,ZERNvv_C0,ZERNrv_C0,ZERNrrv_C0,ZERNrvv_C0,ZERNrrvv_C0] = zernfun(M(k),rC0,vC0);
    [~ ,ZERN_S0,ZERNr_S0,ZERNv_S0,ZERNrr_S0,ZERNvv_S0,ZERNrv_S0,ZERNrrv_S0,ZERNrvv_S0,ZERNrrvv_S0] = zernfun(M(k),rS0,vS0);
    [~ ,ZERN_C1,ZERNr_C1,ZERNv_C1,ZERNrr_C1,ZERNvv_C1,ZERNrv_C1,ZERNrrv_C1,ZERNrvv_C1,ZERNrrvv_C1] = zernfun(M(k),rC1,vC1);
    [~ ,ZERN_S1,ZERNr_S1,ZERNv_S1,ZERNrr_S1,ZERNvv_S1,ZERNrv_S1,ZERNrrv_S1,ZERNrvv_S1,ZERNrrvv_S1] = zernfun(M(k),rS1,vS1);
    
    fun = @(x) f_err(x,Pres,Iota,Psi,bndryR,bndryZ,NFP,M(k),N(k),lm,ln,iM,ZERN_C0,ZERNr_C0,ZERNv_C0,ZERNrr_C0,ZERNvv_C0,ZERNrv_C0,ZERNrrv_C0,ZERNrvv_C0,ZERNrrvv_C0,...
                                                                          ZERN_S0,ZERNr_S0,ZERNv_S0,ZERNrr_S0,ZERNvv_S0,ZERNrv_S0,ZERNrrv_S0,ZERNrvv_S0,ZERNrrvv_S0,...
                                                                          ZERN_C1,ZERNr_C1,ZERNv_C1,ZERNrr_C1,ZERNvv_C1,ZERNrv_C1,ZERNrrv_C1,ZERNrvv_C1,ZERNrrvv_C1,...
                                                                          ZERN_S1,ZERNr_S1,ZERNv_S1,ZERNrr_S1,ZERNvv_S1,ZERNrv_S1,ZERNrrv_S1,ZERNrvv_S1,ZERNrrvv_S1,...
                                                                          rC0,drC0,dvC0,rS0,drS0,dvS0,rC1,drC1,dvC1,rS1,drS1,dvS1,w,symm,squr);
    % solve system of equations and save equilibrium
    tic, xequil = fsolve(fun,xequil,options); toc
    save('xequil.mat','xequil')
    
end
