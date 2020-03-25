from math import *

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import sympy as sym
from sympy import Interval

p_0 = 101325
rho_0 = 1.225
T_0 = 288.15
R = 287.05
gamma = 1.4
alpha = -0.0065
MAC = 2.0569
span = 15.911
x = sym.Symbol('x')

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

def avg(lst):
    return(sum(lst)/len(lst))

def underdamped(x, a, c1, c2, lmda1, lmda2):
    return  a + np.exp(lmda1 * x) * ( c1 * np.sin(lmda2 * x) + c2 * np.cos(lmda2 * x))

def overdamped(x, a, b, c):
    return a + b * np.exp(c * x)

def underdamp(x, a, c1, c2, lmda1, lmda2):
    return  a + np.e**(lmda1 * x) * ( c1 * np.sin(lmda2 * x) + c2 * np.cos(lmda2 * x))

def ctt(a, c1, c2, lmda1, lmda2, length, amp, data_time, data_rate):
    func = a + e**(lmda1 * x) * ( c1 * sym.sin(lmda2 * x) + c2 * sym.cos(lmda2 * x))
    dif = sym.diff(func)
    
    eps = amp/100
    zeros = []
    cut = [0]
    zerosfinal = []
    maxima = []
    t = np.linspace(0,length,10000)
    
    for i in range(len(t)):
        if -eps < dif.subs(x, t[i]) < eps:
            zeros.append(t[i])
    
    for i in range(len(zeros)-1):
        if abs(zeros[i+1]-zeros[i]) > 0.1:
            cut.append(i)
    
    zerosfinal.append(avg(zeros[cut[0]:cut[1]+1]))
    
    for i in range(1,len(cut)-1):
        zerosfinal.append(avg(zeros[cut[i]+1:cut[i+1]+1]))
        
    zerosfinal.append(avg(zeros[cut[-1]+1:]))
        
    for i in range(len(zerosfinal)):
        maxima.append(abs(underdamp(zerosfinal[i], a, c1, c2, lmda1, lmda2)-a))
    
    def function(x, a, b):
        return a * np.exp(b * x)
    
    init_guess = [0, lmda1]
    popt, pcov1 = curve_fit(function, zerosfinal, maxima, p0 = init_guess)
    
    p, q = popt
   
    return [p, q]

def eigv_halftime_sym(a, b, c, l, motion):
    t = np.linspace(0,l,100000)
    t_half = []
    
    y = a + b * np.e**(c * t)
    
    for i in range(len(t)):
        initial_amp = y[0]-a
        amp = y[i]-a
        if initial_amp < 2*amp+0.001 and initial_amp > 2*amp-0.001:
            t_half.append(t[i]-t[0])
            
    lmda1 = (log(0.5)/avg(t_half)) * MAC/V_true[motion]
    return [avg(t_half), lmda1]

def eigv_halftime_asym(a, b, c, l, motion):
    t = np.linspace(0,l,100000)
    t_half = []
    
    y = a + b * np.e**(c * t)
    
    for i in range(len(t)):
        initial_amp = y[0]-a
        amp = y[i]-a
        if initial_amp < 2*amp+0.001 and initial_amp > 2*amp-0.001:
            t_half.append(t[i]-t[0])
            
    lmda1 = (log(0.5)/avg(t_half)) * span/V_true[motion]
    return [avg(t_half), lmda1]

def eigv_period_sym(lmda2, motion):
    P = (2*pi)/abs(lmda2)
    mu = ((2*pi)/P) * MAC/V_true[motion]
    return [P, mu]

def eigv_period_asym(lmda2, motion):
    P = (2*pi)/abs(lmda2)
    mu = ((2*pi)/P) * span/V_true[motion]
    return [P, mu]

short_period = pd.read_excel('short_period.xlsx')
short_period2 = pd.read_excel('short_period2.xlsx')
phugoid = pd.read_excel('phugoid.xlsx')
spiral = pd.read_excel('spiral.xlsx')
dutch_roll = pd.read_excel('dutch_roll.xlsx')
roll_damping = pd.read_excel('roll_damping.xlsx')

sp1_time = short_period['time']-short_period['time'].iloc[0]
sp1_pitch_rate = short_period['bPitchRate']

sp2_time = short_period2['time']-short_period2['time'].iloc[0]
sp2_pitch_rate = short_period2['bPitchRate']

ph_time = phugoid['time']-phugoid['time'].iloc[0]
ph_pitch_rate = phugoid['bPitchRate']

spiral_time = spiral['time']-spiral['time'].iloc[0]
spiral_roll_angle = spiral['Roll_angle']
spiral_yaw_rate = spiral['bYawRate']

dr_time = dutch_roll['time']-dutch_roll['time'].iloc[0]
dr_roll_rate = dutch_roll['bRollRate']
dr_yaw_rate = dutch_roll['bYawRate']

rd_time = roll_damping['time']-roll_damping['time'].iloc[0]
rd_roll_rate = roll_damping['bRollRate']

bound_sp1 = (-1000,1000)
bound_sp2 = (-1000,1000) 
bound_ph = (-1000,1000)
bound_spiral_yaw = ([3, -50, -10], [6, 0, 0])
bound_spiral_roll = ([0, -150, -1], [150, 0, 0])
bound_dr_roll = (-1000,1000)
bound_dr_yaw = (-1000,1000)
bound_rd = ([2, -10, -10], [5, 0, 0])

init_guess_sp1 = []
init_guess_sp2 = [-1.8, 1.2, -5.2]
init_guess_ph = []
init_guess_spiral_yaw = [4, -4.6, -0.11]
init_guess_spiral_roll = [70, -40, -0.02]
init_guess_dr_roll = []
init_guess_dr_yaw = []
init_guess_rd = [2.4, -1.4, -0.5]

sp1_fit_par, pcov1 = curve_fit(underdamped, sp1_time, sp1_pitch_rate, bounds = bound_sp1)
sp2_fit_par, pcov2 = curve_fit(underdamped, sp2_time, sp2_pitch_rate, bounds = bound_sp2)
ph_fit_par, pcov3 = curve_fit(underdamped, ph_time, ph_pitch_rate, bounds = bound_ph)
spiral_roll_fit_par, pcov4 = curve_fit(overdamped, spiral_time, spiral_roll_angle, p0 = init_guess_spiral_roll, bounds = bound_spiral_roll)
spiral_yaw_fit_par, pcov5 = curve_fit(overdamped, spiral_time, spiral_yaw_rate, p0 = init_guess_spiral_yaw, bounds = bound_spiral_yaw)
dr_roll_fit_par, pcov6 = curve_fit(underdamped, dr_time, dr_roll_rate, bounds = bound_dr_roll)
dr_yaw_fit_par, pcov7 = curve_fit(underdamped, dr_time, dr_yaw_rate, bounds = bound_dr_yaw)
rd_fit_par, pcov8 = curve_fit(overdamped, rd_time, rd_roll_rate, p0 = init_guess_rd, bounds = bound_rd)

print(sp1_fit_par, 'overdamp')
print(sp2_fit_par, 'overdamp')
print(ph_fit_par, 'underdamp')
print(spiral_roll_fit_par, 'overdamp')
print(spiral_yaw_fit_par, 'overdamp')
print(dr_roll_fit_par, 'underdamp')
print(dr_yaw_fit_par, 'underdamp')
print(rd_fit_par, 'overdamp')

sp1 = list(sp1_fit_par)
sp2 = list(sp2_fit_par)
ph = list(ph_fit_par)
spiral_roll = list(spiral_roll_fit_par)
spiral_yaw = list(spiral_yaw_fit_par)
dr_roll = list(dr_roll_fit_par)
dr_yaw = list(dr_yaw_fit_par)
rd = list(rd_fit_par)

t_sp1 = np.linspace(0, sp1_time.iloc[-1], 1000)
t_sp2 = np.linspace(0, sp2_time.iloc[-1], 1000)
t_ph = np.linspace(0, ph_time.iloc[-1], 1000)
t_spiral = np.linspace(0, spiral_time.iloc[-1], 1000)
t_dr = np.linspace(0, dr_time.iloc[-1], 1000)
t_rd = np.linspace(0, rd_time.iloc[-1], 1000)

sp1_yfit = underdamped(t_sp1, sp1[0], sp1[1], sp1[2], sp1[3], sp1[4])
sp2_yfit = underdamped(t_sp2, sp2[0], sp2[1], sp2[2], sp2[3], sp2[4])
ph_yfit = underdamped(t_ph, ph[0], ph[1], ph[2], ph[3], ph[4])
spiral_roll_yfit = overdamped(t_spiral, spiral_roll[0], spiral_roll[1], spiral_roll[2])
spiral_yaw_yfit = overdamped(t_spiral, spiral_yaw[0], spiral_yaw[1], spiral_yaw[2])
dr_roll_yfit = underdamped(t_dr, dr_roll[0], dr_roll[1], dr_roll[2], dr_roll[3], dr_roll[4])
dr_yaw_yfit = underdamped(t_dr, dr_yaw[0], dr_yaw[1], dr_yaw[2], dr_yaw[3], dr_yaw[4])
rd_yfit = overdamped(t_rd, rd[0], rd[1], rd[2])

Short_Period1 = ctt(sp1[0], sp1[1], sp1[2], sp1[3], sp1[4], 4, 2, sp1_time, sp1_pitch_rate)
Short_Period2 = ctt(sp2[0], sp2[1], sp2[2], sp2[3], sp2[4], 5, 2, sp2_time, sp2_pitch_rate)
Phugoid = ctt(ph[0], ph[1], ph[2], ph[3], ph[4], 160, 1.5, ph_time, ph_pitch_rate)
Dutch_Roll_Roll = ctt(dr_roll[0], dr_roll[1], dr_roll[2], dr_roll[3], dr_roll[4], 12, 12.5, dr_time, dr_roll_rate)
Dutch_Roll_Yaw = ctt(dr_yaw[0], dr_yaw[1], dr_yaw[2], dr_yaw[3], dr_yaw[4], 12, 17.5, dr_time, dr_yaw_rate)

eigenvalue_short_period_1 = eigv_halftime_sym(0, Short_Period1[0], Short_Period1[1], 4, 0)
eigenvalue_short_period_2 = eigv_halftime_sym(0, Short_Period2[0], Short_Period2[1], 5, 1)
eigenvalue_phugoid = eigv_halftime_sym(0, Phugoid[0], Phugoid[1], 350, 2)
eigenvalue_spiral_roll = eigv_halftime_asym(spiral_roll[0], spiral_roll[1], spiral_roll[2], 30, 3)
eigenvalue_spiral_yaw = eigv_halftime_asym(spiral_yaw[0], spiral_yaw[1], spiral_yaw[2], 30, 3)
eigenvalue_dutch_roll_roll = eigv_halftime_asym(0, Dutch_Roll_Roll[0], Dutch_Roll_Roll[1], 15, 4)
eigenvalue_dutch_roll_yaw = eigv_halftime_asym(0, Dutch_Roll_Yaw[0], Dutch_Roll_Yaw[1], 15, 4)
eigenvalue_roll_damping = eigv_halftime_asym(rd[0], rd[1], rd[2], 1.5, 5)

ev_period_sp1 = eigv_period_sym(sp1[4], 0)
ev_period_sp2 = eigv_period_sym(sp2[4], 1)
ev_period_phugoid = eigv_period_sym(ph[4], 2)
ev_period_dr_roll = eigv_period_asym(dr_roll[4], 4)
ev_period_dr_yaw = eigv_period_asym(dr_yaw[4], 4)

motiondata = [['Short Period 1']  + eigenvalue_short_period_1  + ev_period_sp1,
              ['Short Period 2']  + eigenvalue_short_period_2  + ev_period_sp2,
              ['Phugoid']         + eigenvalue_phugoid         + ev_period_phugoid,
              ['Spiral Roll']     + eigenvalue_spiral_roll,
              ['Spiral Yaw']      + eigenvalue_spiral_yaw,
              ['Dutch Roll Roll'] + eigenvalue_dutch_roll_roll + ev_period_dr_roll,
              ['Dutch Roll Yaw']  + eigenvalue_dutch_roll_yaw  + ev_period_dr_yaw,
              ['Roll Damping']    + eigenvalue_roll_damping]

dataframe = pd.DataFrame(motiondata, columns = ['Type of Motion', 'Half Time', 'Eigenvalue Real', 'Period', 'Eigenvalue Imaginary'])

pd.DataFrame.to_excel(dataframe, 'eigenvalues_of_all_motions.xlsx')

y_decay_shortperiod1 = overdamped(t_sp1, sp1[0], Short_Period1[0], Short_Period1[1])
y_decay_shortperiod2 = overdamped(t_sp2, sp2[0], Short_Period2[0], Short_Period2[1])
y_decay_phugoid = overdamped(t_ph, ph[0], Phugoid[0], Phugoid[1])
y_decay_dr_roll = overdamped(t_dr, dr_roll[0], Dutch_Roll_Roll[0], Dutch_Roll_Roll[1])
y_decay_dr_yaw = overdamped(t_dr, dr_yaw[0], Dutch_Roll_Yaw[0], Dutch_Roll_Yaw[1])

#Short period 1
plt.scatter(sp1_time, sp1_pitch_rate, label='Data')
plt.plot(t_sp1 ,sp1_yfit, 'r-', linewidth=0.75, label = 'Fitted Curve')
plt.plot(t_sp1, y_decay_shortperiod1, 'g-.', linewidth=0.75, label = 'Exponential Decay')
plt.title('Pitch rate during first short period motion')
plt.xlabel('Time [sec]')
plt.ylabel('Pitch Rate [deg/sec]')
plt.legend(loc = 'best')
plt.show()

#Short period 2
plt.scatter(sp2_time, sp2_pitch_rate, label='Data')
plt.plot(t_sp2 ,sp2_yfit, 'r-', linewidth=0.75, label = 'Fitted Curve')
plt.plot(t_sp2, y_decay_shortperiod2, 'g-.', linewidth=0.75, label = 'Exponential Decay')
plt.title('Pitch rate during second short period motion')
plt.xlabel('Time [sec]')
plt.ylabel('Pitch Rate [deg/sec]')
plt.legend(loc = 'best')
plt.show()

#Phugoid
plt.scatter(ph_time, ph_pitch_rate, s=0.75, label='Data')
plt.plot(t_ph, ph_yfit, 'r-', linewidth=0.75, label = 'Fitted Curve')
plt.plot(t_ph, y_decay_phugoid, 'g-.', linewidth=0.75, label = 'Exponential Decay')
plt.title('Pitch rate during phugoid motion')
plt.xlabel('Time [sec]')
plt.ylabel('Pitch Rate [deg/sec]')
plt.legend(loc = 'lower right')
plt.show()

#Spiral
plt.scatter(spiral_time, spiral_roll_angle, s=0.6, label='Data')
plt.plot(t_spiral, spiral_roll_yfit, 'r-', linewidth=0.75, label = 'Fitted Curve')
plt.title('Roll angle during spiral motion')
plt.xlabel('Time [sec]')
plt.ylabel('Roll Angle [deg]')
plt.legend(loc = 'best')
plt.show()

plt.scatter(spiral_time, spiral_yaw_rate, s=0.6, label='Data')
plt.plot(t_spiral, spiral_yaw_yfit, 'r-', linewidth=0.75, label = 'Fitted Curve')
plt.title('Yaw rate during spiral motion')
plt.xlabel('Time [sec]')
plt.ylabel('Yaw Rate [deg/sec]')
plt.legend(loc = 'best')
plt.show()

#Dutch roll
plt.scatter(dr_time, dr_roll_rate, s=1, label='Data')
plt.plot(t_dr, dr_roll_yfit, 'r-', linewidth=0.5, label = 'Fitted Curve')
plt.plot(t_dr, y_decay_dr_roll, 'g-.', linewidth=0.5, label = 'Exponential Decay')
plt.title('Roll rate during Dutch roll motion')
plt.xlabel('Time [sec]')
plt.ylabel('Roll Rate [deg/sec]')
plt.legend(loc = 'best')
plt.show()

plt.scatter(dr_time, dr_yaw_rate, s=1, label='Data')
plt.plot(t_dr, dr_yaw_yfit, 'r-', linewidth=0.5, label = 'Fitted Curve')
plt.plot(t_dr, y_decay_dr_yaw, 'g-.', linewidth=0.5, label = 'Exponential Decay')
plt.title('Yaw rate during Dutch roll motion')
plt.xlabel('Time [sec]')
plt.ylabel('Yaw Rate [deg/sec]')
plt.legend(loc = 'best')
plt.show()

#Roll damping
plt.scatter(rd_time, rd_roll_rate, label='Data')
plt.plot(t_rd, rd_yfit, 'r-', linewidth=0.75, label = 'Fitted Curve')
plt.title('Roll rate during a-periodic roll motion')
plt.xlabel('Time [sec]')
plt.ylabel('Roll Rate [deg/sec]')
plt.legend(loc = 'best')
plt.show()

print('done')