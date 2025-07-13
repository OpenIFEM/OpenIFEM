close all
clear
clc

par.mu       = 1;         
par.rho_f    = 1;
par.mu_s     = 1;
par.lambda_s = 1e16;
par.rho_s    = 1;
p=par;
k   = 1;   % wavenumber
d   = 1e-10;        % overlap width

% digits(100)    
% k = vpa(1);  d = vpa(1);
% p = struct('mu',vpa(1),'rho_f',vpa(100), ...
%            'mu_s',vpa(1),'lambda_s',vpa(1),'rho_s',vpa(1));
% 
% % for loglam = -2:-1:-8              % 1e-2 … 1e-8
% %     lam = vpa(10)^loglam;
% %     %F   = lam^2 * detA_hp(lam,k,d,p);
% %     F   = detA_hp(lam,k,d,p);
% %     % fprintf('λ=1e%2d  |F|/λ^6 = %.15g\n', loglam, double(abs(F)/lam^6));
% %     fprintf('λ=1e%2d  |F|/λ^4 = %.15g\n', loglam, double(abs(F)/lam^4));
% % end

%% compute m
% 
% % fsi-1
% %F = @(lam) lam.^2 .* detA_hp(lam,k,d,p);  
% % omega   = linspace(-0.55,0.55,4000);
% % omega   = omega(abs(omega) > 1e-3);  
% % Faxis  = arrayfun(@(om) F(vpa(1i*om)), omega);
% % m       = min(abs(Faxis));  
% 
% %fsi-2
% %F = @(lam) detA_hp(lam,k,d,p);  
% % omega = linspace(-0.08,0.08,4000);  
% % omega = omega(abs(omega) > 1e-3);      
% % Faxis = arrayfun(@(om) F(vpa(1i*om)), omega);
% % m     = min(abs(Faxis)); 
% 
% %fsi-3
% F = @(lam) detA_hp(lam,k,d,p);  
% omega = linspace(-0.55, 0.55, 4000);      
% omega = omega(abs(omega) > 1e-3);
% Faxis = arrayfun(@(om) F(vpa(1i*om)), omega); 
% m     = min(abs(Faxis));
% fprintf('minimum |detA| on imaginary axis = %.3e\n', double(m));
% 
% 
%% compute M (r=0.05)
% % 
%     Zx = linspace(0,0.25,401); 
%     Zy = linspace(-0.55,0.55,401);
% 
% %  Zx = linspace(0,0.25,401); 
% %  Zy = linspace(-0.08,0.08,401);
% % 
% bot   =  Zx   + 1i*Zy(1);         % bottom  (y = –0.55)
% top   =  Zx   + 1i*Zy(end);       % top     (y =  +0.55)
% left  =  0    + 1i*Zy;            % left    (x =  0)
% left(Zy==0) = [];                 % **remove λ = 0 + 0 i**
% right = 0.25 + 1i*Zy;             % right   (x =  0.25)
% % % 
% boundary = [bot, right, top, left];   
% % % 
% Fedge   = arrayfun(F, boundary);
% % 
% Mstrip  = max(abs(Fedge));
% % 
% % 
% M  = Mstrip / vpa(0.05);
% fprintf('maximum |detA| on the rectangle = %.3e\n', double(Mstrip));
% fprintf('Cauchy derivative bound M       = %.3e\n', double(M));
% 
% 
%%   winding nuber from eps<x<sigma
% 
% %eps = vpa(m/M);
% %eps = 4.34e-13; fsi-1
% sigma = vpa(0.2);
% Ymax = vpa(2);     
% 
% Ny = 1200;  Nx = 120;                     % 600 pts per side (fine)
% 
% ygrid = -Ymax : Ymax/Ny : Ymax;
% ygrid = ygrid(abs(ygrid) > 1e-2);       % remove the noisy centre
% 
% bot   =  eps   + ygrid*1i;              % bottom (x = ε)
% right = ( eps : (sigma-eps)/Nx : sigma ) + Ymax*1i;
% top   =  sigma + fliplr(ygrid)*1i;      % top    (x = σ)
% left  = ( sigma : -(sigma-eps)/Nx : eps ) - Ymax*1i;
% C     = [bot, right, top, left];
% 
% Fvals = arrayfun(F, C);
% 
% fprintf('min |F| on contour = %.2e\n', double(min(abs(Fvals))))
% 
% theta   = imag(log(Fvals)); 
% 
% dtheta  = diff(theta);
% 
% two_pi  = vpa(2)*vpa(pi);
% 
% for j = 1:numel(dtheta)               % manual unwrap (100-digit safe)
%     if dtheta(j) >  pi, dtheta(j) = dtheta(j) - two_pi; end
%     if dtheta(j) < -pi, dtheta(j) = dtheta(j) + two_pi; end
% end
% 
% wN = round(double(sum(dtheta) / (2*pi)));
% 
% fprintf('winding N = %d\n', wN);   


%% solid-only verification
% 
% f = @(lam) det(solidSubMatrix(lam,k,d,par) );
% 
% function As = solidSubMatrix(lam,k,d,par)
%     [~,Af] = detA(lam,k,d,par);                
%     As     = Af([5 7 6 8],[2 3 4 5]);         
% end
% 
% lam_i   = linspace(-2,2, 10001); 
% lam     = 1i*lam_i;             
% 
% detVals = arrayfun(f,lam);
% detReg   = abs(lam).^4 .* abs(detVals);           
% 
% figure('Color','w'),  
% semilogy(lam_i, abs(detVals),'b','LineWidth',3, ...
%          'DisplayName','$|\det A_{\mathrm{solid}}|$');
% hold on
% semilogy(lam_i, abs(detReg) ,'r--','LineWidth',3, ...
%           'DisplayName','$|\lambda|^{4}\det A_{\mathrm{solid}}$');
% 
% xlabel('$\mathrm{Im}(\lambda)=\omega$','Interpreter','latex','FontSize',40)
% ylabel('magnitude (log scale)','Interpreter','latex','FontSize',40)
% grid on, box on
% legend('Interpreter','latex','FontSize',32,'Location','best')
% set(gca,'FontSize',32)
% title('Fourth-order pole removed – red curve is flat','FontSize',36)


% figure
% semilogy(lam_i , abs(detVals), 'LineWidth', 3), grid on
% xlabel('$\mathrm{Im}(\lambda)$','FontWeight','bold','Interpreter','latex',FontSize=40)
% ylabel('$|\det A_\mathrm{solid}(i\omega)|$','FontWeight','bold','Interpreter','latex',FontSize=40)
% ax = gca;
% ax.FontSize = 40;
% grid on
% legend('$|\det A_{\mathrm{solid}}(i\omega)|$', ...
%        'Interpreter', 'latex', ...
%        'FontSize', 40, ...          
%        'Location', 'best', ...
%        'Box', 'off');


% % clear,  format long g
% % k = 1; d = 1;
% % par.mu = 1; par.rho_f = 1;          
% % par.mu_s = 1; par.lambda_s = 1; par.rho_s = 1;
% % det4 = @(lam)                            ...
% %    det( solidSubMatrix(lam,k,d,par) );
% % 
% % seeds = [ 0.41  1.0  1.62  1.732]; 
% % 
% % F  = @(w) [ real(det4(1i*w)); imag(det4(1i*w)) ];
% % opts = optimoptions('fsolve','Algorithm','levenberg-marquardt','Display','final-detailed','TolX',1e-16);
% % 
% % roots = [];
% % for s = seeds
% %     try
% %         wroot = fsolve(F, s, opts);
% %         if wroot > 1e-3                     
% %             roots(end+1) = wroot;       
% %         end
% %     catch
% %         ignore failures (bad seed)
% %     end
% % end
% % 
% % roots = unique(round(roots,8));           
% % disp('Roots of det A_solid (λ = i ω) for k = d = 1:')
% % fprintf('   ω = %.5f\n', roots)
% % 


%% show that -nu*k^2 is indeed a branch point for pure fluid
% nu  = par.mu/par.rho_f; 
% lambda0 = -nu*k^2; 
% dlist  = logspace(-4,-14,7);      
% 
% detvals = zeros(size(dlist));
% sigma_min = zeros(size(dlist));
% 
% for j = 1:numel(dlist)
%     d = dlist(j);
% 
%     [~, A] = detA(lambda0, k, d, par);   
%     detvals(j)   = det(A);              
%     s            = svd(A,'econ');        
%     sigma_min(j) = s(end);               
% end
% 
% loglog(dlist,abs(detvals),'o-',LineWidth=3); hold on
% loglog(dlist,sigma_min,'x-',LineWidth=3); 
% %legend('$|detA(-\nu k^2)|$','$\sigma_{\mathrm{min}}$', 'interpreter','latex','FontWeight','bold')
% legend('$|detA(-\nu k^2)|$', 'interpreter','latex','FontWeight','bold')
% %xlabel('$\mathrm{Im}(\lambda)$','FontWeight','bold','Interpreter','latex',FontSize=20)
% xlabel('Overlapping region thickness','FontSize',40)
% ylabel('|detA| (log scale)',FontSize=40)
% ax = gca;
% ax.FontSize = 40;
% grid on
% legend('FontSize', 40, ...           
%        'Location', 'best', ...
%        'Box', 'off');
% 
% 
% figure
% loglog(dlist,abs(detvals)./dlist.^2,'o-'), grid on
% hold on
% plot(dlist,sigma_min./dlist,'o-'), 
% xlabel('Overlapping region thickness','FontSize',40)
% ylabel('|detA|/d^2',FontSize=40)
% ax = gca;
% ax.FontSize = 40;
% grid on
% legend('FontSize', 40, ...           
%        'Location', 'best', ...
%        'Box', 'off');



%%  check there is no eigenvalues for Re(lambda)>0

% nu  = par.mu/par.rho_f; 
% lam_max = 100*nu*k^2;         
% N       = 20000;         
% lamVec  = linspace(0,lam_max,N);   
% detVec  = zeros(1,N);
% 
% for n = 1:N
%     lambda          = lamVec(n);
%     detVec(n)       = detA(lambda,k,d,p);  
% end
% 
% 
% figure
% 
% semilogy(lamVec,abs(detVec),'LineWidth',3,DisplayName='absolute value of det(A(\lambda))')
% xlabel('\lambda  (real)'), ylabel('|det A|  (log scale)')
% grid on
% ax = gca;
% ax.FontSize = 40;
% grid on
% legend('FontSize', 40, ...           
%        'Location', 'best', ...
%        'Box', 'off');
% 
% figure
% 
% plot(lamVec,sign(real(detVec)),'LineWidth',3,DisplayName='sign')
% xlabel('\lambda  (real)'), ylabel('Sign of the Re of det(A(\lambda))')
% %title('No sign change  ⇒  no eigen-root on Re(\lambda)>0')
% ylim([-1.5 1.5]), grid on
% ax = gca;
% ax.FontSize = 40;
% grid on
% legend('FontSize', 40, ...          
%        'Location', 'best', ...
%        'Box', 'off');


%%  Pure-fluid branch-point check  (d = 1e-10 ≈ “no solid”)
% nu = p.mu/p.rho_f;
% lamStar = -nu*k^2;         
% 
% lamVec   = linspace(-100, -1, 40000);
% 
% detAbs   = zeros(size(lamVec));
% detScal  = detAbs;          
% sigmaMin = detAbs;
% 
% for j = 1:numel(lamVec)
%     lam = lamVec(j);
%     [~,A] = detA(lam,k,d,p);
% 
% 
%         T = eye(8);
%         T(2,2) = lam;       
%         T(3,3) = lam;      
%         A = A * T;         
% 
%     detAbs(j)  = abs(det(A));              
%     detScal(j) = detAbs(j) ./ abs(lam)^6;   
% end
% 

% semilogy(lamVec,detAbs ,'b' ,'LineWidth',3,'DisplayName','|detA|'); hold on
% % semilogy(lamVec,detScal,'r--','LineWidth',3,'DisplayName','|detA| / |λ|^{6}');
% hxl = xline(lamStar,'k:','\lambda = -\nu k^{2}','LineWidth',2,...
%       'LabelOrientation','horizontal','LabelVerticalAlignment','middle','HandleVisibility', 'off');
% % Set font size of the xline label
% hxl.FontSize = 40;
% xlabel('\lambda  (real axis)', 'FontSize', 40)
% %yline(eps,'k--','\epsilon_{mach}','HandleVisibility','off')
% ylabel('|detA|  (log scale)')
% % title('Red curve is flat ⇒ only branch point at λ = -νk^{2}')
% ax = gca;
% ax.FontSize = 40;
% grid on
% legend('FontSize', 40, ...           
%        'Location', 'best', ...
%        'Box', 'off');
% 

% figure('Color','w')                                 
% semilogy(lamVec, detAbs, 'b-', 'LineWidth', 3, 'DisplayName', 'det(A(\lambda))'); hold on
% 
% hLine = xline(lamStar, 'k:', '\lambda = -\nu k^2', ...
%       'LineWidth', 2, ...                          
%       'LabelOrientation', 'horizontal', ...
%       'LabelVerticalAlignment', 'middle', ...
%       'FontSize', 40);
% 
% hLine.Annotation.LegendInformation.IconDisplayStyle = 'off';
% 
% xlabel('\lambda  (real axis)', 'FontSize', 40)
% ylabel('|det A|',               'FontSize', 40)
% 
% set(gca, 'FontSize', 40)    
% grid on;  box on
% legend(Box="off")









