
close all; clear;  clc;
detfun  = @detA;    
%detfun  = @detA_dummy;
par.mu       = 1;         
par.rho_f    = 100;
par.mu_s     = 1;
par.lambda_s = 1;
par.rho_s    = 1;
p=par;
k   = 1;   
d   = 1;       


NyList = [100 200 400];        % initial points per contour
RList  = [1 2 4];               % initial half-widths
sigma0 = 0.2;                   % fixed left- or right-wall location

% tblNL = NaN(numel(NyList),numel(RList));   % winding number  (left)
 %tblNR = NaN(size(tblNL));                  % winding number  (right)

 tblNR = NaN(numel(NyList),numel(RList));   % winding number  (left)

% tblRL = NaN(size(tblNL));                  % final R  (left run)
 tblRR = NaN(size(tblNR));                  % final R  (right run)

% tblNyL = NaN(size(tblNL));                 % final Ny (left run)
    tblNyR = NaN(size(tblNR));                 % final Ny (right run)             

optBase = struct('makePlot',false,'epsDet',1e-8,'sigma0',sigma0);


for i = 1:numel(NyList)
for j = 1:numel(RList)

    opts       = optBase;
    opts.Ny0   = NyList(i);
    opts.R0    = RList(j);

    %infoL      = countHalfPlane(detfun,k,d,p,'left', opts);
    infoR      = countHalfPlane(detfun,k,d,p,'right',opts);

    % tblL(i,j)  = infoL.N;
    tblR(i,j)  = infoR.N;

    
    % tblRL(i,j)  = infoL.R;   tblRR(i,j)  = infoR.R;
    % tblNyL(i,j) = infoL.Ny;  tblNyR(i,j) = infoR.Ny;

    tblRR(i,j)  = infoR.R;
    tblNyR(i,j) = infoR.Ny;

  %   fprintf(['Ny0 = %4d  R0 = %3.1f   →  ', ...
  %            'NL = %2d (R_f = %4.1f, Ny_f = %5d)   ', ...
  %            'NR = %2d (R_f = %4.1f, Ny_f = %5d)   ', ...
  %            'min|det| = %.1e\n'], ...
  %           opts.Ny0, opts.R0, ...
  %           infoL.N, infoL.R, infoL.Ny, ...
  %           infoR.N, infoR.R, infoR.Ny, ...
  %           min(infoL.minDet,infoR.minDet));

        fprintf(['Ny0 = %4d  R0 = %3.1f   →  ', ...
                 'NR = %2d (R_f = %4.1f, Ny_f = %5d)   ', ...
                 'min|det| = %.1e\n'], ...
                opts.Ny0, opts.R0, ...
                infoR.N, infoR.R, infoR.Ny, ...
                min(infoR.minDet));
end
end
