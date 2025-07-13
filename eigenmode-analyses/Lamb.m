clear, clc, format long g


k  = 1;                
d  = 1;  h = d/2;       
ct = 1;                
cl = sqrt(3);          


alpha = @(w)  sqrt( k^2 - (w/cl).^2 );         
beta  = @(w)  sqrt( k^2 - (w/ct).^2 );        


FA = @(w) tanh(beta(w).*h) ./ tanh(alpha(w).*h) ...
        - ( (k^2 + beta(w).^2).^2 ./ (4*alpha(w).*beta(w)*k^2) );


FS_sub = @(w) tanh(beta(w).*h) ./ tanh(alpha(w).*h) ...
            - 4*alpha(w).*beta(w)*k^2 ./ (k^2 + beta(w).^2).^2;

q      = @(w) sqrt(w.^2 - ct^2);               
FS_sup = @(w) tan( q(w).*h ) ./ tanh(alpha(w).*h) ...
            - 4*alpha(w).*q(w)*k^2 ./ (k^2 - q(w).^2).^2;


opts = optimset('TolX',1e-12,'Display','off');


wA0 = fzero(FA, [0.3 0.6], opts);           


wS0 = ct * k;                                
wP  = cl * k;                                 


wScan   = linspace(wS0+1e-4, wP-1e-4, 4001);    
vals    = FS_sup(wScan);
ix      = find( vals(1:end-1).*vals(2:end) < 0, 1 );   
wS1     = fzero(FS_sup, [wScan(ix) , wScan(ix+1)], opts);


fprintf('\nLamb roots for k = d = 1  (λ = i ω)\n')
fprintf('  A0  (antisym)  : ω = %.15f\n', wA0)
fprintf('  S0  (shear)    : ω = %.15f\n', wS0)
fprintf('  S1  (symm)     : ω = %.15f\n', wS1)
fprintf('  P-wave cutoff  : ω = %.15f\n\n', wP)

