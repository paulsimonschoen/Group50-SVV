from math import *

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import sympy as sym

p_0 = 101325
rho_0 = 1.225
T_0 = 288.15
R = 287.05
gamma = 1.4
alpha = -0.0065
c = 2.0569

def avg(lst):
    return(sum(lst)/len(lst))    

data = [['short_period_1', 198, -7.75, 6816.36],
        ['short_period_2', 160, -8.5 , 7424   ],
        ['phugoid'       , 160, -8.5 , 7422.69],
        ['spiral'        , 162, -9   , 7827.27],
        ['dutch_roll'    , 161, -9.75, 7888.02],
        ['roll_damping'  , 179, -9   , 7587.14]]

df = pd.DataFrame(data, columns = ['Name', 'V_true', 'T_static', 'Pressure_height'])

T_static_K = list(df['T_static']+273.15)
V_true = list(df['V_true']*0.514444444)
H_p = list(df['Pressure_height']*0.3048)

V_EAS = []

for i in range(len(data)):
    p = p_0*(1+(alpha*H_p[i])/T_0)**(-(9.81/(alpha*R)))
    rho = p/(R*T_static_K[i])
    V_eas = V_true[i]*sqrt(rho/rho_0)
    
    V_EAS.append(V_eas)

def eigv_halftime(a, b, c, l, motion):
    t = np.linspace(0,l,10000)
    t_half = []
    
    y = a + b * np.e**(c * t)
    
    for i in range(len(t)):
        initial_amp = y[0]-a
        amp = y[i]-a
        if initial_amp < 2*amp+0.001 and initial_amp > 2*amp-0.001:
            t_half.append(t[i]-t[0])
            
    lmda1 = (log(0.5)/avg(t_half)) * c/V_EAS[motion]
    print(lmda1)
    print(avg(t_half))
    return(lmda1)

def eigv_period(lmda2, motion):
    P = (2*pi)/abs(lmda2)
    mu = ((2*pi)/P) * c/V_EAS[motion]
    print(P)
    print(mu)
    return mu

eigv_halftime(0, 15.523439750332592, -0.2321767072020265, 15, 4)

#eigv_period(-1.85898071, 4)

    #print(lmda1)
    #print(avg(t_half))