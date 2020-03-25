from math import *

import pandas as pd
import matplotlib.pyplot as plt

short_period = pd.read_excel('type_of_motion/short_period.xlsx')
short_period2 = pd.read_excel('type_of_motion/phugoid.xlsx')
phugoid = pd.read_excel('type_of_motion/phugoid.xlsx')
spiral = pd.read_excel('type_of_motion/dutch_roll.xlsx')
dutch_roll = pd.read_excel('type_of_motion/aperiodic_roll.xlsx')
roll_damping = pd.read_excel('type_of_motion/aperiodic_roll.xlsx')

short_period.drop(short_period.head(5970).index,inplace=True)
short_period.drop(short_period.tail(4550).index,inplace=True)

short_period2.drop(short_period2.head(470).index,inplace=True)
short_period2.drop(short_period2.tail(1825).index,inplace=True)

phugoid.drop(phugoid.head(460).index,inplace=True)
phugoid.drop(phugoid.tail(1400).index,inplace=True)

spiral.drop(spiral.head(30).index,inplace=True)
spiral.drop(spiral.tail(470).index,inplace=True)

dutch_roll.drop(dutch_roll.head(110).index,inplace=True)
dutch_roll.drop(dutch_roll.tail(520).index,inplace=True)

roll_damping.drop(roll_damping.head(704).index,inplace=True)
roll_damping.drop(roll_damping.tail(50).index,inplace=True)
    
x1 = short_period['time']-2091
y1 = short_period['bPitchRate']
a1 = short_period['d_e']
b1 = short_period['Det']

x2 = phugoid['time']-2595
y2 = phugoid['bPitchRate']
a2 = phugoid['d_e']
b2 = phugoid['Det']

x3 = spiral['time']-2865
y3 = spiral['bYawRate']
z3 = spiral['Roll_angle']
a3 = spiral['d_a']
b3 = spiral['d_r']

x4 = dutch_roll['time']-2792.5
y4 = dutch_roll['bRollRate']
z4 = dutch_roll['bYawRate']
a4 = dutch_roll['d_a']
b4 = dutch_roll['d_r']

x5 = roll_damping['time']-2852
y5 = roll_damping['bRollRate']
z5 = roll_damping['Roll_angle']
a5 = roll_damping['d_a']
b5 = roll_damping['d_r']

x6 = short_period2['time']-2596
y6 = short_period2['bPitchRate']
a6 = short_period2['d_e']
b6 = short_period2['Det']


#short period 1
fig1, axs1 = plt.subplots(2, 1, constrained_layout=True)
axs1[0].scatter(x1,y1,s=0.6)
axs1[0].set_title('Pitch rate response')
axs1[0].set_xlabel('Time [sec]')
axs1[0].set_ylabel('Pitch rate [deg/sec]')
fig1.suptitle('Control surface deflection for first short period motion', fontsize=16)

axs1[1].scatter(x1,a1, s=0.6)
axs1[1].set_title('Step input of elevator')
axs1[1].set_xlabel('Time [sec]')
axs1[1].set_ylabel('Elevator deflection [deg]')
plt.show()

#short period 2
fig2, axs2 = plt.subplots(2, 1, constrained_layout=True)
axs2[0].scatter(x6,y6,s=0.6)
axs2[0].set_title('Pitch rate response')
axs2[0].set_xlabel('Time [sec]')
axs2[0].set_ylabel('Pitch rate [deg/sec]')
fig2.suptitle('Control surface deflection for second short period motion', fontsize=16)

axs2[1].scatter(x6,a6, s=0.6)
axs2[1].set_title('Step input of elevator')
axs2[1].set_xlabel('Time [sec]')
axs2[1].set_ylabel('Elevator deflection [deg]')
plt.show()

#phugoid
fig3, axs3 = plt.subplots(2, 1, constrained_layout=True)
axs3[0].scatter(x2,y2,s=0.6)
axs3[0].set_title('Pitch rate response')
axs3[0].set_xlabel('Time [sec]')
axs3[0].set_ylabel('Pitch rate [deg/sec]')
fig3.suptitle('Control surface deflection for phugoid motion', fontsize=16)

axs3[1].scatter(x2,a2, s=0.6)
axs3[1].set_title('Step input of elevator')
axs3[1].set_xlabel('Time [sec]')
axs3[1].set_ylabel('Elevator deflection [deg]')
plt.show()

#spiral
fig4, axs4 = plt.subplots(2, 1, constrained_layout=True)
axs4[0].scatter(x3,y3,s=0.6)
axs4[0].set_title('Roll rate')
axs4[0].set_xlabel('Time [sec]')
axs4[0].set_ylabel('Roll rate [deg/sec]')
fig4.suptitle('Spiral motion', fontsize=16)

axs4[1].scatter(x3,z3, s=0.6)
axs4[1].set_title('Roll angle')
axs4[1].set_xlabel('Time [sec]')
axs4[1].set_ylabel('Roll angle [deg]')
plt.show()

#dutch roll
fig5, axs5 = plt.subplots(2, 1, constrained_layout=True)
axs5[0].scatter(x4,y4,s=0.6)
axs5[0].set_title('Roll rate response')
axs5[0].set_xlabel('Time [sec]')
axs5[0].set_ylabel('Roll rate [deg/sec]')
fig5.suptitle('Control surface deflection for Dutch roll', fontsize=16)

axs5[1].scatter(x4,b4, s=0.6)
axs5[1].set_title('Rudder input')
axs5[1].set_xlabel('Time [sec]')
axs5[1].set_ylabel('Rudder deflection [deg]')
plt.show()

#roll damping
fig6, axs6 = plt.subplots(2, 1, constrained_layout=True)
axs6[0].scatter(x5,y5,s=0.6)
axs6[0].set_title('Roll rate response')
axs6[0].set_xlabel('Time [sec]')
axs6[0].set_ylabel('Roll rate [deg/sec]')
fig6.suptitle('Control surface deflection for a-periodic roll', fontsize=16)

axs6[1].scatter(x5,a5, s=0.6)
axs6[1].set_title('Step input of aileron')
axs6[1].set_xlabel('Time [sec]')
axs6[1].set_ylabel('Aileron deflection [deg]')
axs6[1].tick_params(axis='y', labelcolor='blue')
'''
fig7, axs7 = plt.subplots(2, 1, constrained_layout=True)
axs7[0].scatter(x5,y5,s=0.6)
axs7[0].set_title('Roll rate response')
axs7[0].set_xlabel('Time [sec]')
axs7[0].set_ylabel('Roll rate [deg/sec]')
fig7.suptitle('Control surface deflection for a-periodic roll', fontsize=16)

axs7[1].scatter(x5,b5, s=0.6)
axs7[1].set_title('Step input of rudder')
axs7[1].set_xlabel('Time [sec]')
axs7[1].set_ylabel('Rudder deflection [deg]')
'''
axs8 = axs6[1].twinx()
axs8.scatter(x5,b5, s=0.6, c='red')
axs8.set_ylabel('Rudder deflection [deg]')
axs8.tick_params(axis='y', labelcolor='red')
plt.show()

'''
pd.DataFrame.to_excel(short_period, 'short_period.xlsx')
pd.DataFrame.to_excel(short_period2, 'short_period2.xlsx')
pd.DataFrame.to_excel(phugoid, 'phugoid.xlsx')
pd.DataFrame.to_excel(spiral, 'spiral.xlsx')
pd.DataFrame.to_excel(dutch_roll, 'dutch_roll.xlsx')
pd.DataFrame.to_excel(roll_damping, 'roll_damping.xlsx')
'''
print('done')