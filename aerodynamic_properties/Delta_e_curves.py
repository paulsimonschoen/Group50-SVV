# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:53:20 2020

@author: Friso
"""
#from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#General properties
K = 273.15
T_0 = 15+K
p_0 = 101325
rho_0 = 1.225
R = 287.05
gamma = 1.4
dT_h = -0.0065
g = 9.08665
M = 0.3
j = 1.4
#Aircraft Dimensions 
S = 30          #Wing area in [m^2]
mac = 2.0569    #Mean aerodynamic chord in [m]
b = 15.911      #Span in [m]
Ws = 60500     #Standard aircraft weight [N]
m_fs = 0.048     #Standard engine fuel flow in [kg/sec]
W_f = 4050      #Fuel Weight in [lbs]
W_emp = 9165    #Empty Weight in [lbs]
Arm_emp = 291.65 #inch
D = 0.693           #m
#Aerodynamic coefficients
C_D_0 = 0.04    #
C_L_a = 5.084   #
e = 0.8         #Oswald factor
C_m_tc = -0.0064
C_m_d = -0.0918
#Getting the data
trim_curve_data = pd.read_excel("Trim_curve_data.xlsx", sheet_name = 'trim_ref')         #Import Excel File with all the data
trim_curve_data.columns = ['Altitude_ft', 'IAS', 'a', 'de', 'detr', 'Fe', 'FFL_lbs/hr', 'FFR_lbs/hr', 'F_used', 'TAT_C']
seat_data = pd.read_excel("Seat_data.xlsx", sheet_name = 'seat_ref')         #Import Excel File with all the data
seat_data.columns = ['Seat_1', 'Seat_2', 'W_pass']
#cg_data = pd.read_excel("cg_data.xlsx")
#cg_data.columns = ['Altitude_ft', 'IAS', 'a', 'de', 'detr', 'Fe', 'FFL_lbs/hr', 'FFR_lbs/hr', 'F_used', 'TAT_C']
thrust_ref = pd.read_excel("Thrust_input_data.xlsx", sheet_name = 'RefT')         #Import Excel File with all the data
thrust_ref.columns = ['T_l', 'T_r']                 #Thrust in [N]
thrust_refs = pd.read_excel("Thrust_input_data.xlsx", sheet_name = 'RefTs')         #Import Excel File with all the data
thrust_refs.columns = ['T_l', 'T_r']                #Thrust in [N]

#Putting the data into different arrays
altitude_m = np.asarray(trim_curve_data['Altitude_ft']*0.3048)  #Altitude in [m]
V_ias = np.asarray(trim_curve_data['IAS']*0.514444444)          #Speed in [m/s]
alpha = np.asarray(trim_curve_data['a'])                        #Angle of Attack in [deg]
delta_e = np.asarray(trim_curve_data['de'])                     #Elevator deflection in [deg]
delta_e_tr = np.asarray(trim_curve_data['detr'])                #Elevator Trim Tab in [deg]
F_e = np.asarray(trim_curve_data['Fe'])                         #Stick force in [N]
ffl = np.asarray(trim_curve_data['FFL_lbs/hr']*0.45359237/3600) #Fuel flow left Engine in [kg/s]
ffr = np.asarray(trim_curve_data['FFR_lbs/hr']*0.45359237/3600) #Fuel flow right Engine in [kg/s]
F_used = np.asarray(trim_curve_data['F_used']*0.45359237)
T_tot_K = np.asarray(trim_curve_data['TAT_C'] + K)              #Total Temperature in [K]
d_T = T_tot_K-(T_0 + dT_h*altitude_m)                           #Change in Temp in [K/m]
W_pass = np.asarray(seat_data['W_pass'])                     #Passenger weight in [kg]
#seat1 = np.asarray(seat_data['Seat_1']*0.0254)
#seat2 = np.asarray(seat_data['Seat_2']*0.0254)
W_tot = ((W_f + W_emp)*0.45359237 + np.sum(W_pass))*g
T_t = np.asarray(thrust_ref['T_l']) + np.asarray(thrust_ref['T_r'])
T_ts = np.asarray(thrust_refs['T_l']) + np.asarray(thrust_refs['T_r'])

#Calculations
p_2 = p_0 * ((1+(dT_h*altitude_m/T_0))**(-g/(dT_h*R)))
M_2 = np.sqrt((2/(j-1))*((1+(p_0/p_2)*((1+(j-1)/(2*j)*(rho_0/p_0)*V_ias ** 2) ** (j/(j-1)) - 1)) **((j-1)/j) - 1))
T_2 = T_tot_K/(1+(gamma-1)/2*M_2**2)
a_2 = np.sqrt(gamma*R*T_2)
V_t2 = M_2*a_2
rho_2 = p_2/(R*T_2)
V_e2 = V_t2*np.sqrt(rho_2/rho_0)
W_2 = W_tot - F_used*g
V_re2 = V_e2*np.sqrt(Ws/W_2)
F_es = F_e *(Ws/W_2)
dT = T_2 - T_tot_K
#print(altitude_m, M_2, dT, ffl, ffr)

T_c = T_t/(0.5*rho_2*V_re2**2*2*D**2)
T_cs = T_ts/(0.5 * rho_2 * V_re2**2 * 2 * D**2)
d_re = delta_e - (1/C_m_d)*C_m_tc*(T_cs-T_c)
#print(d_re)

#Plots
#de/Vre-plot with regression line
poly_fit_d = np.poly1d(np.polyfit(V_re2, delta_e, 2))
plt.figure(1)
xx = np.linspace(np.amin(V_re2), np.amax(V_re2), 100)
plt.plot(xx, poly_fit_d(xx), c='cornflowerblue',linestyle='-', label = "Regression Line")
plt.scatter(V_re2, delta_e, c='navy', label = "Reduced elevator deflection")
plt.xlabel('$\~V_e [m/s]$') # naming the x axis 
plt.ylabel('$\delta_e^* [deg]$') # naming the y axis 
plt.title('Reference Data:  $\delta_e^*$ - $\~V_e$ curve') # giving a title to my graph 
plt.legend() # show a legend on the plot
plt.grid(b=True, which='major', color='lightgray', linestyle='-')
plt.show() # function to show the plot

#Fe/Vre-plot with regression line
poly_fit_f = np.poly1d(np.polyfit(V_re2, F_es, 2))
plt.figure(2)
xx = np.linspace(np.amin(V_re2), np.amax(V_re2), 100)
plt.plot(xx, poly_fit_f(xx), c='cornflowerblue',linestyle='-', label = "Regression Line")
plt.scatter(V_re2, F_es, c='navy', label = "Reduced elevator control force")
plt.xlabel('$\~V_e [m/s]$') # naming the x axis 
plt.ylabel('$F_e^* [N]$') # naming the y axis 
plt.title('Reference Data:  $F_e^*$ - $\~V_e$ curve') # giving a title to my graph 
plt.legend() # show a legend on the plot
plt.grid(b=True, which='major', color='lightgray', linestyle='-')
plt.show() # function to show the plot

#d_delta/d_alpha
#poly_fit_da = np.poly1d(np.polyfit(alpha, delta_e, 1))
#print("Linear equation d_delta/d_alpha =", poly_fit_da)
#plt.figure(3)
#plt.scatter(alpha, delta_e)
#plt.show()