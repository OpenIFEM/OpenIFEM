clear
clc

k = 1;  d = 1;

p.rho_f = 100;
p.mu  = 1;
p.rho_s = 1;
p.lambda_s = 1;
p.mu_s  = 1;


detReg = @(lam) double( detA_hp(lam,k,d,p) / lam.^4 );   
%detReg = @(lam) double( detA_hp(lam,k,d,p)  );   

rootfun = @(x) [ real(detReg(x(1)+1i*x(2)));
                 imag(detReg(x(1)+1i*x(2))) ];


guess = [ -p.mu*k*tanh(k*d)/p.rho_f*d ,  2e-4 ];        
opts  = optimoptions('fsolve', ...
        'Algorithm','trust-region-dogleg', ...
        'Display','iter', ...
        'FunctionTolerance',1e-12, ...
        'StepTolerance',1e-12, ...
        'FiniteDifferenceType','central', ...
        'FiniteDifferenceStepSize',1e-6);    

[xSol,~,flag] = fsolve(rootfun, guess, opts);

lambdaStar = xSol(1) + 1i*xSol(2);
fprintf('\nλ* ≈ %.8f %+7.2e i   |det| = %.2e   (flag %d)\n', ...
        real(lambdaStar), imag(lambdaStar), ...
        abs(detA_hp(lambdaStar,k,d,p)), flag);

[val,A] = detA_hp(lambdaStar,k,d,p);   
s       = svd(A);
sigmaMin = s(end);                    
fprintf('σ_min(A) = %.3e\n',sigmaMin);

F  = @(lam) detA_hp(lam,k,d,p) ./ lam.^4;

fprintf('|F(1.01 λ̂)| = %.3e\n', abs(F(1.01*lambdaStar)));
fprintf('|F(0.99 λ̂)| = %.3e\n', abs(F(0.99*lambdaStar)));
fprintf('|F(λ̂)     | = %.3e\n', abs(F(lambdaStar)));

