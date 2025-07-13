function val = detA_dummy(lambda,~,~,~)


    
    z = [ ...
        -0.6 + 2i ;               
        -3 - 1i ;               
        %1.0      ;             
        -50 + 0.01i ;            
        (-6+2i) * ones(2,1) ;   
        -8 + 3i ; -8 - 3i ;     
          20 + 10i   ;   
        -1+10^-4i;
        -1-10^-4i;
        -1+2*10^-4i;
    ];
  

    val = ones(size(lambda));
    for r = z.'
        val = val .* (lambda - r);
    end
end