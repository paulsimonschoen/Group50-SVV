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

Data = pd.read_excel("../Stationary_measurements.xlsx")
FlightData = pd.read_excel('../reading flight data/FD_AOA.xlsx')
Data.columns = ['time', 'Altitude_ft', 'IAS', 'AOA', 'FFL_lbs/hr', 'FFR_lbs/hr', 'F_used', 'TAT_C']

altitude_m = np.asarray(Data['Altitude_ft']*0.3048)
AOA = np.asarray(Data['AOA']*(pi/180))
V_c = np.asarray(Data['IAS']*0.514444444)
ffl = np.asarray(Data['FFL_lbs/hr']*0.45359237/3600)
ffr = np.asarray(Data['FFR_lbs/hr']*0.45359237/3600)
F_used = np.asarray(Data['F_used']*0.45359237)
T_tot_K = np.asarray(Data['TAT_C'] + K)

Time = []
th0 = []
m = []
v_true = []
v_eas = []
T_static = []

for i in range(len(Data)):
    lst = Data['time'][i].split(':')
    time = float(lst[0])*60+float(lst[1])
    Time.append(time)

    idx = FlightData[FlightData['time']==time].index.values[0]
    th0.append(FlightData['Pitch_angle'][idx]*(pi/180))
    

for i in range(len(Data)): 
    p = p_0*(1+(alpha*altitude_m[i])/T_0)**(-(9.81/(alpha*R)))
    M = sqrt((2/(gamma-1))*((1+(p_0/p)*((1+((gamma-1)/(2*gamma))*(rho_0/p_0)*(V_c[i]**2))**(gamma/(gamma-1))-1))**((gamma-1)/gamma)-1))
    T = T_tot_K[i]/(1+((gamma-1)/2)*(M**2))
    a = (gamma*R*T)**0.5
    V_true = M*a
    rho = p/(R*T)
    V_eas = V_true*sqrt(rho/rho_0)
    
    m.append(M)
    v_true.append(V_true)
    v_eas.append(V_eas)
    T_static.append(T)
    
T_stat = sum(T_static)/len(T_static)
d_T = T_stat-(T_0 + alpha*altitude_m)
print(T_stat)

row_1 = str(altitude_m[0]) + ' ' + str(m[0]) + ' ' + str(d_T[0]) + ' ' + str(ffl[0]) + ' ' + str(ffr[0])
row_2 = str(altitude_m[1]) + ' ' + str(m[1]) + ' ' + str(d_T[1]) + ' ' + str(ffl[1]) + ' ' + str(ffr[1])
row_3 = str(altitude_m[2]) + ' ' + str(m[2]) + ' ' + str(d_T[2]) + ' ' + str(ffl[2]) + ' ' + str(ffr[2])
row_4 = str(altitude_m[3]) + ' ' + str(m[3]) + ' ' + str(d_T[3]) + ' ' + str(ffl[3]) + ' ' + str(ffr[3])
row_5 = str(altitude_m[4]) + ' ' + str(m[4]) + ' ' + str(d_T[4]) + ' ' + str(ffl[4]) + ' ' + str(ffr[4])
row_6 = str(altitude_m[5]) + ' ' + str(m[5]) + ' ' + str(d_T[5]) + ' ' + str(ffl[5]) + ' ' + str(ffr[5])

print(row_1)

f = open('matlab.dat', 'w')
f.write(row_1 + '\n' + row_2 + '\n' + row_3 + '\n' + row_4 + '\n' + row_5 + '\n' + row_6)

df = pd.DataFrame([Time, list(altitude_m), AOA, th0, m, v_true, v_eas, list(ffl), list(ffr), list(F_used)]).T
df.columns = ['time', 'Altitude_m', 'AOA', 'th0', 'mach', 'V_true', 'V_eas', 'FFL', 'FFR', 'Fuel_used']

pd.DataFrame.to_excel(df, '../converted_data.xlsx')

print('done')