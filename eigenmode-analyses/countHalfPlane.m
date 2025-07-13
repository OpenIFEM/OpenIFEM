function info = countHalfPlane(detfun, k, d, p, side, opts)

if nargin < 5 || isempty(side), side = 'left'; end
if nargin<6, opts = struct; end

opts = set_defaults(opts,...
     'R0',4,'sigma0',0.2,'Ny0',1024,'growR',2.0,'growNy',2,...
     'epsDet',1e-8,'dphiMax',pi/1.3,'maxOuter',20,'makePlot',false);

switch lower(side(1))
    case 'l', sgn = -1;      
    case 'r', sgn = +1;     
    otherwise, error('side must be ''left'' or ''right''.');
end

R     = opts.R0;
sigma = opts.sigma0;
Ny    = opts.Ny0;
Nprev      = Inf;
changeSeen = false;   
outer = 0;
sameCnt = 0;    

while true
    outer = outer + 1;
      if outer > opts.maxOuter
          break
      end
    refineCnt = 0;

    while true
       C = halfRectangle(R,sigma,Ny,sgn);

        F = arrayfun(@(lam) lam.^2 .* detfun(lam,k,d,p), C);
        %F = arrayfun(@(lam) detfun(lam,k,d,p), C);
        
        dphi   = angle( F([2:end 1])./F );
        mf    = min(abs(F));

            onWall = abs(real(C) - sgn*sigma) < 1e-12;
            
            minWall  = min(abs(F(onWall)));  

            if any(abs(F(onWall)) < opts.epsDet)
            sigma     = sigma / 2;
            Ny        = opts.Ny0;
            refineCnt = 0;
            continue
            end
              
        if mf > opts.epsDet && all(abs(dphi) <= opts.dphiMax)
            break
        end

        refineCnt = refineCnt + 1;
       
       if refineCnt >= 3
            R         = opts.growR * R;
            Ny        = opts.Ny0;
            refineCnt = 0;           
            continue
        else
            Ny = opts.growNy * Ny;
            continue
        end
    end
    raw = sum(dphi)/(2*pi);

    N = round( raw );

     if N == Nprev
        sameCnt = sameCnt + 1;
    else
        sameCnt = 0;
        if isfinite(Nprev),  changeSeen = true;  end
     end


  if (changeSeen && sameCnt >= 3) || (N == 0 && sameCnt >= 3) 
        break
  end

    Nprev = N;
    R     = opts.growR * R;

end

info = struct('N',N,'R',R,'Ny',Ny,'sigma',sigma,...
              'minDet',mf,'minWall',minWall,'argsum',sum(dphi));
end

function S = set_defaults(S,varargin)
for k = 1:2:numel(varargin)
    if ~isfield(S,varargin{k}),  S.(varargin{k}) = varargin{k+1}; end
end
end


function C = halfRectangle(R,sigma,N,sgn)

t = linspace(0,1,N).';
if sgn>0               
    bottom =  sigma + (R-sigma)*t       - 1i*R;
    right  =  R                         + 1i*(-R + 2*R*t);
    top    =  R     - (R-sigma)*t       + 1i*R;
    left   =  sigma                     + 1i*( R - 2*R*t);
else                   
    bottom = -R       + (R-sigma)*t     - 1i*R;
    right  = -sigma                     + 1i*(-R + 2*R*t);
    top    = -sigma   - (R-sigma)*t     + 1i*R;
    left   = -R                        + 1i*( R - 2*R*t);
end
C = [bottom; right(2:end); top(2:end); left(2:end)];
end


