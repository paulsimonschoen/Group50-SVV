from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

Data = pd.read_excel('../converted_data.xlsx', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
m_pax = pd.read_excel('../weight_addition.xlsx', sheet_name = 'Passengers')
m_fuel = pd.read_excel('../weight_addition.xlsx', sheet_name = 'Fuel')
Thrust = open('../Thrust calculation/thrust.dat', 'r').read()
m_pax.columns = ['Passenger', 'mass']
m_fuel.columns = ['fuel_type', 'mass']

lst = Thrust.split('\n')
thrust = []
for i in range(len(lst)-1):
    Lst = lst[i].split('\t')
    thrust.append(float(Lst[0])+float(Lst[1]))

m_empty = 9165*0.45359237

# Aircraft geometry
    
S      = 30.00	          # wing area [m^2]
Sh     = 0.2 * S         # stabiliser area [m^2]
Sh_S   = Sh / S	          # [ ]
lh     = 0.71 * 5.968    # tail length [m]
c      = 2.0569	          # mean aerodynamic cord [m]
lh_c   = lh / c	          # [ ]
b      = 15.911	          # wing span [m]
bh     = 5.791	          # stabilser span [m]
A      = b ** 2 / S      # wing aspect ratio [ ]
Ah     = bh ** 2 / Sh    # stabilser aspect ratio [ ]
Vh_V   = 1	          # [ ]
ih     = -2 * pi / 180   # stabiliser angle of incidence [rad]
    
# Constant values concerning atmosphere and gravity
    
rho0   = 1.2250          # air density at sea level [kg/m^3] 
lmda = -0.0065         # temperature gradient in ISA [K/m]
Temp0  = 288.15          # temperature at sea level in ISA [K]
R      = 287.05          # specific gas constant [m^2/sec^2K]
g      = 9.81            # [m/sec^2] (gravity constant)

mu_0 = 1.716E-5
T_static   = 262.7861280356529
T_0 = 273.15
suth = 110.4

Cl = []
Cd = []
alpha = []
Reynolds = []
Mach = []

for i in range(len(Data)):
    hp0    = Data['Altitude_m'][i] # pressure altitude in the stationary flight condition [m]
    V0     = Data['V_true'][i]     # true airspeed in the stationary flight condition [m/sec]
    alpha0 = Data['AOA'][i]        # angle of attack in the stationary flight condition [rad]
    th0    = Data['AOA'][i]            # pitch angle in the stationary flight condition [rad]
    M      = Data['mach'][i]

    # Aircraft mass
    m      = m_empty + sum(m_pax['mass']) + sum(m_fuel['mass'])*0.45359237 - Data['Fuel_used'][i] # mass [kg]
    
    # air density [kg/m^3]  
    rho    = rho0 * ((1+(lmda * hp0 / Temp0)))**(-((g / (lmda*R)) + 1))
    W      = m * g            # [N]       (aircraft weight)
    D      = thrust[i]
    
    mu = mu_0*(T_static/T_0)**(3/2)*((T_0+suth)/(T_static+suth))
    Re = (rho*V0*c)/mu
    Reynolds.append(Re)
    Mach.append(M)
    
    # Constant values concerning aircraft inertia
    
    muc    = m / (rho * S * c)
    mub    = m / (rho * S * b)
    KX2    = 0.019
    KZ2    = 0.042
    KXZ    = 0.002
    KY2    = 1.25 * 1.114
    
    # Lift and drag coefficient
    
    CL = 2 * W / (rho * V0 ** 2 * S)              # Lift coefficient [ ]
    CD = 2 * D / (rho * V0 ** 2 * S)
    Cl.append(CL)
    Cd.append(CD)
    alpha.append(alpha0)

print(Reynolds)
print(Mach)

X = pd.DataFrame(alpha).values.reshape(-1,1)
Y = pd.DataFrame(Cl).values.reshape(-1,1)
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

CLa = linear_regressor.coef_                         # Slope of CL-alpha curve [ ]

plt.scatter(alpha, Cl)
plt.title('Cl-alpha')
plt.ylabel('lift coeficient [-]')
plt.xlabel('angle of attack [rad]')

plt.plot(alpha, Y_pred, color='red')
plt.show()

plt.scatter(Cd, Cl)
plt.title('Cl-Cd')
plt.ylabel('lift coeficient')
plt.xlabel('drag coeficient')
plt.show()

plt.scatter(alpha, Cd)
plt.title('Cd-alpha')
plt.ylabel('drag coeficient')
plt.xlabel('angle of attack')
plt.show()

Cl_check = []
for i in range(len(Cl)):
    Cl_check.append(Cl[i]**2)

U = pd.DataFrame(Cl_check).values.reshape(-1,1)
V = pd.DataFrame(Cd).values.reshape(-1,1)
linear_regressor = LinearRegression()
linear_regressor.fit(U, V)
V_pred = linear_regressor.predict(U)



ax1 = plt.subplot(221)
plt.scatter(alpha, Cl)
plt.plot(alpha, Y_pred, color='red', linewidth = 0.5)
plt.title('$Cl-\u03B1$', fontsize = 10)
plt.ylabel('Lift coeficient $Cl$ [$-$]', fontsize = 7)
plt.xlabel('Angle of attack $\u03B1$ [$rad$]', fontsize = 7)
plt.setp(ax1.get_xticklabels(), fontsize = 7)
plt.setp(ax1.get_yticklabels(), fontsize = 7)
plt.grid(True)

ax2 = plt.subplot(222)
plt.scatter(alpha, Cd)
plt.title('$Cd-\u03B1$', fontsize = 10)
plt.ylabel('Drag coeficient $Cd$ [$-$]', fontsize = 7)
plt.xlabel('Angle of attack $\u03B1$ [$deg$]', fontsize = 7)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.setp(ax2.get_xticklabels(), fontsize = 7)
plt.setp(ax2.get_yticklabels(), fontsize = 7)
plt.grid(True)

ax3 = plt.subplot(223)
plt.scatter(Cd, Cl)
plt.title('$Cl-Cd$', fontsize = 10)
plt.ylabel('Lift coeficient $Cl$ [$-$]', fontsize = 7)
plt.xlabel('Drag coeficient $Cd$ [$-$]', fontsize = 7)
plt.setp(ax3.get_xticklabels(), fontsize = 7)
plt.setp(ax3.get_yticklabels(), fontsize = 7)
plt.grid(True)

ax4 = plt.subplot(224)
plt.scatter(Cl_check, Cd)
plt.plot(Cl_check, V_pred, color='red', linewidth = 0.5)
plt.title('$Cl^2-Cd$', fontsize = 10)
plt.ylabel('Drag coefficient $Cd$ [$-$]', fontsize = 7)
plt.xlabel('Lift coefficient squared $Cl^2$ [$-$]', fontsize = 7)
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
plt.setp(ax4.get_xticklabels(), fontsize = 7)
plt.setp(ax4.get_yticklabels(), fontsize = 7)
plt.grid(True)

plt.suptitle('First stationary measurements', fontsize=16)

plt.subplots_adjust(wspace=0.1, hspace=0.5)
plt.savefig('../reading flight data/Graphs/stationary_measurements.png', dpi = 300)
plt.show()

CD0 = linear_regressor.intercept_                  # Zero lift drag coefficient [ ]
e = 1/(linear_regressor.coef_[0]*pi*A)             # Oswald factor [ ]

'''    
# Longitudinal stability
Cma    =             # longitudinal stabilty [ ]
Cmde   =             # elevator effectiveness [ ]

# Aerodynamic constants
    
Cmac   = 0                      # Moment coefficient about the aerodynamic centre [ ]
CNwa   = CLa                    # Wing normal force slope [ ]
CNha   = 2 * pi * Ah / (Ah + 2) # Stabiliser normal force slope [ ]
depsda = 4 / (A + 2)            # Downwash gradient [ ]
'''

df = pd.DataFrame([CLa, CD0, e]).T
df.columns = ['CLa [rad]', 'CD0', 'e']

pd.DataFrame.to_excel(df, '../Stability_coefficients.xlsx')

print('done')