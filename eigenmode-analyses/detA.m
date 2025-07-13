
function [value,A] = detA(lambda,k,d,p)

beta_f = sqrt(k^2 + lambda*p.rho_f/p.mu);
if real(beta_f)<0
    beta_f = -beta_f; 
end

gamma  = sqrt(k^2 + p.rho_s*lambda.^2/p.mu_s);

if real(gamma )<0
    gamma  = -gamma ; 
end

chi    = sqrt(k^2 + p.rho_s*lambda.^2/(p.lambda_s+2*p.mu_s));
 
if real(chi  )<0  
    chi    = -chi ; 
end

cg  = cosh(gamma*d);  
sg  = sinh(gamma*d);
cchi= cosh(chi*d);
schi= sinh(chi*d);

k2      = k^2;
DK       = (p.lambda_s+2*p.mu_s)*chi^2 - p.lambda_s*k2;
beta2k2  = beta_f^2 + k2;    
gamma2k2 = gamma^2  + k2; 

 A = zeros(8,8);

% row 1: v_x at x=0
 A(1,1) =  1i*k;
 A(1,2) = -1i*k;
 A(1,5) = -lambda*chi;
 
 % row 2: v_x at x=d
  A(2,6) =  1i*k;
  A(2,2) = -1i*k*cg;
  A(2,3) = -1i*k*sg;
  A(2,4) = -lambda*chi*schi;
  A(2,5) = -lambda*chi*cchi;

   % row 3: v_y at x=0
   A(3,1) = -beta_f;
   A(3,4) = -lambda*1i*k;
   A(3,3) =  gamma;  

   % row 4: v_y at x=d
    A(4,6) =  beta_f;
    A(4,2) =  gamma*sg;
    A(4,3) =  gamma*cg;
    A(4,4) = -lambda*1i*k*cchi;
    A(4,5) = -lambda*1i*k*schi;

    % row 5: sigma_xx at x=0 
    A(5,1) =  2*p.mu*1i*k*beta_f;
    A(5,3) =  -2*p.mu_s*1i*k*gamma / lambda;          
    A(5,4) = -DK;
    A(5,7) = -1;
    
    %row 6: sigma_xx at x=d
    A(6,6) = -2*p.mu*1i*k*beta_f;
    A(6,2) = -2*p.mu_s*1i*k*gamma / lambda * sg;
    A(6,3) = -2*p.mu_s*1i*k*gamma / lambda * cg;
    A(6,4) = -DK*cchi;
    A(6,5) = -DK*schi;
    A(6,8) = -1;

    % row 7: shear traction at x=0
    A(7,1) = -p.mu * beta2k2;
    A(7,2) =  p.mu_s * gamma2k2 / lambda; 
    A(7,5) =  -2*p.mu_s*1i*k*chi; 
 
   % row 8: shear traction at x=d
    A(8,2) = p.mu_s * gamma2k2 / lambda * cg;
    A(8,3) = p.mu_s * gamma2k2 / lambda * sg;  
    A(8,6) = -p.mu * beta2k2;  
    A(8,4) =  -2*p.mu_s*1i*k*chi*schi;  
    A(8,5) =  -2*p.mu_s*1i*k*chi*cchi;  
   
value = det(A);

end




