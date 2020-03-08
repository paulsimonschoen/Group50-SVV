from math import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

K = 273.15
T_0 = 15+K
p_0 = 101325
rho_0 = 1.225
R = 287.05
gamma = 1.4
alpha = -0.0065
M=0.3
d = {}

Data = pd.read_excel("Thrust_input_data.xlsx")
Data.columns = ['Altitude_ft', 'IAS', 'FFL_lbs/hr', 'FFR_lbs/hr', 'TAT_C']

altitude_m = np.asarray(Data['Altitude_ft']*0.3048)
V_eas = np.asarray(Data['IAS']*0.514444444)
ffl = np.asarray(Data['FFL_lbs/hr']*0.45359237/3600)
ffr = np.asarray(Data['FFR_lbs/hr']*0.45359237/3600)
T_tot_K = np.asarray(Data['TAT_C'] + K)
d_T = T_tot_K-(T_0 + alpha*altitude_m)

m = list(np.arange(0.1,0.9,0.0001))
m_eq = []
mtest = []

for i in range(len(Data)):   
    a = []
    V_tas = []
    
    for M in m:
        A = (gamma*R*(T_tot_K[i]/(1+((gamma-1)/2)*(M**2))))**0.5
        V = V_eas[i]*(1+((gamma-1)/2)*(M**2))**(1/(2*(gamma-1)))
        
        a.append(A)
        V_tas.append(V)
    
    for j in range(len(a)):
        if (m[j]-0.00005)<(V_tas[j]/a[j]) and (m[j]+0.00005)>(V_tas[j]/a[j]):
            m_eq.append(round(m[j],10))

row_1 = str(altitude_m[0]) + ' ' + str(m_eq[0]) + ' ' + str(d_T[0]) + ' ' + str(ffl[0]) + ' ' + str(ffr[0])
row_2 = str(altitude_m[1]) + ' ' + str(m_eq[1]) + ' ' + str(d_T[1]) + ' ' + str(ffl[1]) + ' ' + str(ffr[1])
row_3 = str(altitude_m[2]) + ' ' + str(m_eq[2]) + ' ' + str(d_T[2]) + ' ' + str(ffl[2]) + ' ' + str(ffr[2])
row_4 = str(altitude_m[3]) + ' ' + str(m_eq[3]) + ' ' + str(d_T[3]) + ' ' + str(ffl[3]) + ' ' + str(ffr[3])
row_5 = str(altitude_m[4]) + ' ' + str(m_eq[4]) + ' ' + str(d_T[4]) + ' ' + str(ffl[4]) + ' ' + str(ffr[4])
row_6 = str(altitude_m[5]) + ' ' + str(m_eq[4]) + ' ' + str(d_T[5]) + ' ' + str(ffl[5]) + ' ' + str(ffr[5])

f = open('matlab.dat', 'w')
f.write(row_1 + '\n' + row_2 + '\n' + row_3 + '\n' + row_4 + '\n' + row_5 + '\n' + row_6)

print('done')