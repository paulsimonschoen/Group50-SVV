# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:53:20 2020

@author: Friso
"""
#from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

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
m_fs = 0.04     #Standard engine fuel flow in [kg/sec]
W_f = 2700      #Fuel Weight in [lbs]
W_emp = 9165    #Empty Weight in [lbs]
#Aerodynamic coefficients
C_D_0 = 0.04    #
C_L_a = 5.084   #
e = 0.8         #Oswald factor
C_m_tc = -0.0064
#Getting the data
trim_curve_data = pd.read_excel("Trim_curve_data.xlsx")         #Import Excel File with all the data
trim_curve_data.columns = ['Altitude_ft', 'IAS', 'a', 'de', 'detr', 'Fe', 'FFL_lbs/hr', 'FFR_lbs/hr', 'F_used', 'TAT_C']
seat_data = pd.read_excel("Seat_data.xlsx")         #Import Excel File with all the data
seat_data.columns = ['Seat_1', 'Seat_2', 'W_pass']
thrust_input_data = pd.read_excel("Thrust_input_data.xlsx")         #Import Excel File with all the data
thrust_input_data.columns = ['T_l', 'T_r']

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
W_pass_kg = np.asarray(seat_data['W_pass'])                     #Passenger weight in [kg]
seat1 = np.asarray(seat_data['Seat_1']*0.0254)
seat2 = np.asarray(seat_data['Seat_2']*0.0254)
W_tot = ((W_f + W_emp)*0.45359237 + np.sum(W_pass_kg))*g

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

#Plots
#poly_reg = PolynomialFeatures(degree=2)
#X_poly = poly_reg.fit_transform(V_re2.reshape(-1, 1))
#poly_reg_2 = PolynomialFeatures(degree=3)
#X_poly_2 = poly_reg_2.fit_transform(V_re2.reshape(-1, 1))
#lin_reg_3 = LinearRegression()
#lin_reg_3.fit(X_poly_2, delta_e.reshape(-1, 1))
#y_pred = lin_reg_3.predict(X_poly)
plt.figure(1);
plt.scatter(V_re2, delta_e);
#plt.plot(V_re2, y_pred);
plt.show()
#print(r2_score(y, y_pred))

plt.show()
plt.figure(2)
plt.scatter(V_re2, F_es)
plt.show()