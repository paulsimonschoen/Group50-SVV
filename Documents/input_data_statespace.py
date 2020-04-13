hp0    = 2077.525944      	      
V0     = 95.51           
alpha0 = 0.075452           
th0    = 0.068771           

m      = 5874,33462383          

e      = 0.8778            
CD0    = 0.024           
CLa    = 5.88      

Cma    = -0.62554           
Cmde   = -1.304


S      = 30.00	     
Sh     = 0.2 * S      
Sh_S   = Sh / S	       
lh     = 0.71 * 5.968   
c      = 2.0569	         
lh_c   = lh / c	         
b      = 15.911	       
bh     = 5.791	        
A      = b ^ (2 / S)    
Ah     = bh ^ (2 / Sh)   
Vh_V   = 1	        
ih     = -2 * pi / 180  

rho0   = 1.2250         
lambda = -0.0065        
Temp0  = 288.15         
R      = 287.05        
g      = 9.81            
  
rho    = rho0 * power( ((1+(lambda * hp0 / Temp0))), (-((g / (lambda*R)) + 1)))   
W      = m * g          

muc    = m / (rho * S * c)
mub    = m / (rho * S * b)
KX2    = 0.019
KZ2    = 0.042
KXZ    = 0.002
KY2    = 1.25 * 1.114

Cmac   = 0                     
CNwa   = CLa                   
CNha   = 2 * pi * Ah / (Ah + 2)
depsda = 4 / (A + 2)           

CL = 2 * W / (rho * V0 ^ 2 * S)             
CD = CD0 + (CLa * alpha0) ^ 2 / (pi * A * e)

CX0    = W * sin(th0) / (0.5 * rho * V0 ^ 2 * S)
CXu    = -0.02792
CXa    = +0.47966		
CXadot = +0.08330
CXq    = -0.28170
CXde   = -0.03728

CZ0    = -W * cos(th0) / (0.5 * rho * V0 ^ 2 * S)
CZu    = -0.37616
CZa    = -5.74340
CZadot = -0.00350
CZq    = -5.66290
CZde   = -0.69612

Cmu    = +0.06990
Cmadot = +0.17800
Cmq    = -8.79415

CYb    = -0.7500
CYbdot =  0     
CYp    = -0.0304
CYr    = +0.8495
CYda   = -0.0400
CYdr   = +0.2300

Clb    = -0.10260
Clp    = -0.71085
Clr    = +0.23760
Clda   = -0.23088
Cldr   = +0.03440

Cnb    =  +0.1348
Cnbdot =   0     
Cnp    =  -0.0602
Cnr    =  -0.2061
Cnda   =  -0.0120
Cndr   =  -0.0939
