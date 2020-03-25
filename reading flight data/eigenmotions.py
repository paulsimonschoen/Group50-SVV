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
short_period.drop(short_period.tail(4400).index,inplace=True)

short_period2.drop(short_period2.head(470).index,inplace=True)
short_period2.drop(short_period2.tail(1750).index,inplace=True)

phugoid.drop(phugoid.head(250).index,inplace=True)
phugoid.drop(phugoid.tail(300).index,inplace=True)

spiral.drop(spiral.head(30).index,inplace=True)
spiral.drop(spiral.tail(470).index,inplace=True)

dutch_roll.drop(dutch_roll.head(110).index,inplace=True)
dutch_roll.drop(dutch_roll.tail(520).index,inplace=True)

roll_damping.drop(roll_damping.head(704).index,inplace=True)
roll_damping.drop(roll_damping.tail(0).index,inplace=True)

#Short Period 1
sp1_initV = [short_period['V_true'].iloc[0]]
sp1_initAOA = [short_period['AOA'].iloc[0]]
sp1_initPA = [short_period['Pitch_angle'].iloc[0]]
sp1_initPR = [short_period['bPitchRate'].iloc[0]]

t1 = short_period['time']-short_period['time'].iloc[0]
h1 = short_period['alt']
v1 = short_period['V_true']
aoa1 = short_period['AOA']
pr1 = short_period['bPitchRate']
pa1 = short_period['Pitch_angle']
d_e1 = short_period['d_e']
Det1 = short_period['Det']

axsp1 = plt.subplot(321)
plt.plot(t1, h1)
plt.ylabel('Altitude [ft]', fontsize = 5)
plt.setp(axsp1.get_xticklabels(), visible = False)
plt.setp(axsp1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axsp2 = plt.subplot(322)
plt.plot(t1, pa1)
plt.ylabel('Pitch angle [deg]', fontsize = 5)
plt.setp(axsp2.get_xticklabels(), visible = False)
axsp2.yaxis.tick_right()
plt.setp(axsp2.get_yticklabels(), fontsize = 7)
axsp2.yaxis.set_label_position("right")
plt.grid(True)

axsp3 = plt.subplot(323, sharex=axsp1)
plt.plot(t1, v1)
plt.ylabel('True airspeed [V]', fontsize = 5)
plt.setp(axsp3.get_xticklabels(), visible = False)
plt.setp(axsp3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axsp4 = plt.subplot(324, sharex=axsp2)
plt.plot(t1, pr1)
plt.ylabel('Pitch rate [deg/sec]', fontsize = 5)
plt.setp(axsp4.get_xticklabels(), visible = False)
axsp4.yaxis.tick_right()
plt.setp(axsp4.get_yticklabels(), fontsize = 7)
axsp4.yaxis.set_label_position("right")
plt.grid(True)

axsp5 = plt.subplot(325, sharex=axsp1)
plt.plot(t1, aoa1)
plt.ylabel('Angle of attack [deg]', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axsp5.get_xticklabels(), fontsize = 7)
plt.setp(axsp5.get_yticklabels(), fontsize = 7)
plt.grid(True)

axsp6 = plt.subplot(326, sharex=axsp2)
plt.plot(t1, d_e1)
plt.ylabel('Elevator deflection [deg]', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axsp6.get_xticklabels(), fontsize = 7)
axsp6.yaxis.tick_right()
plt.setp(axsp6.get_yticklabels(), fontsize = 7)
axsp6.yaxis.set_label_position("right")
plt.grid(True)

plt.suptitle('First short Period motion', fontsize=16)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig('Graphs/short_period1.png', dpi = 300)
plt.show()

#Short Period 2
sp2_initV = [short_period2['V_true'].iloc[0]]
sp2_initAOA = [short_period2['AOA'].iloc[0]]
sp2_initPA = [short_period2['Pitch_angle'].iloc[0]]
sp2_initPR = [short_period2['bPitchRate'].iloc[0]]

t2 = short_period2['time']-short_period2['time'].iloc[0]
h2 = short_period2['alt']
v2 = short_period2['V_true']
aoa2 = short_period2['AOA']
pr2 = short_period2['bPitchRate']
pa2 = short_period2['Pitch_angle']
d_e2 = short_period2['d_e']
Det2 = short_period2['Det']

axs1 = plt.subplot(321)
plt.plot(t2, h2)
plt.ylabel('Altitude [ft]', fontsize = 5)
plt.setp(axs1.get_xticklabels(), visible = False)
plt.setp(axs1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axs2 = plt.subplot(322)
plt.plot(t2, pa2)
plt.ylabel('Pitch angle [deg]', fontsize = 5)
plt.setp(axs2.get_xticklabels(), visible = False)
axs2.yaxis.tick_right()
plt.setp(axs2.get_yticklabels(), fontsize = 7)
axs2.yaxis.set_label_position("right")
plt.grid(True)

axs3 = plt.subplot(323, sharex=axs1)
plt.plot(t2, v2)
plt.ylabel('True airspeed [V]', fontsize = 5)
plt.setp(axs3.get_xticklabels(), visible = False)
plt.setp(axs3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axs4 = plt.subplot(324, sharex=axs2)
plt.plot(t2, pr2)
plt.ylabel('Pitch rate [deg/sec]', fontsize = 5)
plt.setp(axs4.get_xticklabels(), visible = False)
axs4.yaxis.tick_right()
plt.setp(axs4.get_yticklabels(), fontsize = 7)
axs4.yaxis.set_label_position("right")
plt.grid(True)

axs5 = plt.subplot(325, sharex=axs1)
plt.plot(t2, aoa2)
plt.ylabel('Angle of attack [deg]', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axs5.get_xticklabels(), fontsize = 7)
plt.setp(axs5.get_yticklabels(), fontsize = 7)
plt.grid(True)

axs6 = plt.subplot(326, sharex=axs2)
plt.plot(t2, d_e2)
plt.ylabel('Elevator deflection [deg]', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axs6.get_xticklabels(), fontsize = 7)
axs6.yaxis.tick_right()
plt.setp(axs6.get_yticklabels(), fontsize = 7)
axs6.yaxis.set_label_position("right")
plt.grid(True)

plt.suptitle('Second short period motion', fontsize=16)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig('Graphs/short_period2.png', dpi = 300)
plt.show()

#Phugoid
ph_initV = [phugoid['V_true'].iloc[0]]
ph_initAOA = [phugoid['AOA'].iloc[0]]
ph_initPA = [phugoid['Pitch_angle'].iloc[0]]
ph_initPR = [phugoid['bPitchRate'].iloc[0]]

t3 = phugoid['time']-phugoid['time'].iloc[0]
h3 = phugoid['alt']
v3 = phugoid['V_true']
aoa3 = phugoid['AOA']
pr3 = phugoid['bPitchRate']
pa3 = phugoid['Pitch_angle']
d_e3 = phugoid['d_e']
Det3 = phugoid['Det']

axph1 = plt.subplot(321)
plt.plot(t3, h3)
plt.ylabel('Altitude [ft]', fontsize = 5)
plt.setp(axph1.get_xticklabels(), visible = False)
plt.setp(axph1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axph2 = plt.subplot(322)
plt.plot(t3, pa3)
plt.ylabel('Pitch angle [deg]', fontsize = 5)
plt.setp(axph2.get_xticklabels(), visible = False)
axph2.yaxis.tick_right()
plt.setp(axph2.get_yticklabels(), fontsize = 7)
axph2.yaxis.set_label_position("right")
plt.grid(True)

axph3 = plt.subplot(323, sharex=axph1)
plt.plot(t3, v3)
plt.ylabel('True airspeed [V]', fontsize = 5)
plt.setp(axph3.get_xticklabels(), visible = False)
plt.setp(axph3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axph4 = plt.subplot(324, sharex=axph2)
plt.plot(t3, pr3)
plt.ylabel('Pitch rate [deg/sec]', fontsize = 5)
plt.setp(axph4.get_xticklabels(), visible = False)
axph4.yaxis.tick_right()
plt.setp(axph4.get_yticklabels(), fontsize = 7)
axph4.yaxis.set_label_position("right")
plt.grid(True)

axph5 = plt.subplot(325, sharex=axph1)
plt.plot(t3, aoa3)
plt.ylabel('Angle of attack [deg]', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axph5.get_xticklabels(), fontsize = 7)
plt.setp(axph5.get_yticklabels(), fontsize = 7)
plt.grid(True)

axph6 = plt.subplot(326, sharex=axph2)
plt.plot(t3, d_e3)
plt.ylabel('Elevator deflection [deg]', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axph6.get_xticklabels(), fontsize = 7)
axph6.yaxis.tick_right()
plt.setp(axph6.get_yticklabels(), fontsize = 7)
axph6.yaxis.set_label_position("right")
plt.grid(True)

plt.suptitle('Phugoid motion', fontsize=16)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig('Graphs/phugoid.png', dpi = 300)
plt.show()

#Spiral
spiral_initYA = [0]
spiral_initRA = [spiral['Roll_angle'].iloc[0]]
spiral_initRR = [spiral['bRollRate'].iloc[0]]
spiral_initYR = [spiral['bYawRate'].iloc[0]]

t4 = spiral['time']-spiral['time'].iloc[0]
h4 = spiral['alt']
v4 = spiral['V_true']
rr4 = spiral['bRollRate']
yr4 = spiral['bYawRate']
ra4 = spiral['Roll_angle']
d_a4 = spiral['d_a']
d_r4 = spiral['d_r']

axsr1 = plt.subplot(321)
plt.plot(t4, h4)
plt.ylabel('Altitude [ft]', fontsize = 5)
plt.setp(axsr1.get_xticklabels(), visible = False)
plt.setp(axsr1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axsr2 = plt.subplot(322)
plt.plot(t4, ra4)
plt.ylabel('Roll angle [deg]', fontsize = 5)
plt.setp(axsr2.get_xticklabels(), visible = False)
axsr2.yaxis.tick_right()
plt.setp(axsr2.get_yticklabels(), fontsize = 7)
axsr2.yaxis.set_label_position("right")
plt.grid(True)

axsr3 = plt.subplot(323, sharex=axsr1)
plt.plot(t4, v4)
plt.ylabel('True airspeed [V]', fontsize = 5)
plt.setp(axsr3.get_xticklabels(), visible = False)
plt.setp(axsr3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axsr4 = plt.subplot(324, sharex=axsr2)
plt.plot(t4, rr4)
plt.ylabel('Roll rate [deg/sec]', fontsize = 5)
plt.setp(axsr4.get_xticklabels(), visible = False)
axsr4.yaxis.tick_right()
plt.setp(axsr4.get_yticklabels(), fontsize = 7)
axsr4.yaxis.set_label_position("right")
plt.grid(True)

axsr5 = plt.subplot(325, sharex=axsr1)
plt.plot(t4, d_a4, label = 'Aileron deflection')
plt.plot(t4, d_r4, label = 'Rudder deflection')
plt.ylabel('Pilot input', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axsr5.get_xticklabels(), fontsize = 7)
plt.setp(axsr5.get_yticklabels(), fontsize = 7)
plt.legend(loc = 'best', prop={'size': 5})
plt.grid(True)

axsr6 = plt.subplot(326, sharex=axsr2)
plt.plot(t4, yr4)
plt.ylabel('Yaw rate [deg/sec]', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axsr6.get_xticklabels(), fontsize = 7)
axsr6.yaxis.tick_right()
plt.setp(axsr6.get_yticklabels(), fontsize = 7)
axsr6.yaxis.set_label_position("right")
plt.grid(True)

plt.suptitle('Spiral motion', fontsize=16)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig('Graphs/spiral.png', dpi = 300)
plt.show()

#Dutch Roll
dr_initYA = [0]
dr_initRA = [dutch_roll['Roll_angle'].iloc[0]]
dr_initRR = [dutch_roll['bRollRate'].iloc[0]]
dr_initYR = [dutch_roll['bYawRate'].iloc[0]]

t5 = dutch_roll['time']-dutch_roll['time'].iloc[0]
h5 = dutch_roll['alt']
v5 = dutch_roll['V_true']
rr5 = dutch_roll['bRollRate']
yr5 = dutch_roll['bYawRate']
ra5 = dutch_roll['Roll_angle']
d_a5 = dutch_roll['d_a']
d_r5 = dutch_roll['d_r']

axdr1 = plt.subplot(321)
plt.plot(t5, h5)
plt.ylabel('Altitude [ft]', fontsize = 5)
plt.setp(axdr1.get_xticklabels(), visible = False)
plt.setp(axdr1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axdr2 = plt.subplot(322)
plt.plot(t5, ra5)
plt.ylabel('Roll angle [deg]', fontsize = 5)
plt.setp(axdr2.get_xticklabels(), visible = False)
axdr2.yaxis.tick_right()
plt.setp(axdr2.get_yticklabels(), fontsize = 7)
axdr2.yaxis.set_label_position("right")
plt.grid(True)

axdr3 = plt.subplot(323, sharex=axdr1)
plt.plot(t5, v5)
plt.ylabel('True airspeed [V]', fontsize = 5)
plt.setp(axdr3.get_xticklabels(), visible = False)
plt.setp(axdr3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axdr4 = plt.subplot(324, sharex=axdr2)
plt.plot(t5, rr5)
plt.ylabel('Roll rate [deg/sec]', fontsize = 5)
plt.setp(axdr4.get_xticklabels(), visible = False)
axdr4.yaxis.tick_right()
plt.setp(axdr4.get_yticklabels(), fontsize = 7)
axdr4.yaxis.set_label_position("right")
plt.grid(True)

axdr5 = plt.subplot(325, sharex=axdr1)
plt.plot(t5, d_a5, label = 'Aileron deflection')
plt.plot(t5, d_r5, label = 'Rudder deflection')
plt.ylabel('Pilot input', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axdr5.get_xticklabels(), fontsize = 7)
plt.setp(axdr5.get_yticklabels(), fontsize = 7)
plt.legend(loc = 'best', prop={'size': 5})
plt.grid(True)

axdr6 = plt.subplot(326, sharex=axdr2)
plt.plot(t5, yr5)
plt.ylabel('Yaw rate [deg/sec]', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axdr6.get_xticklabels(), fontsize = 7)
axdr6.yaxis.tick_right()
plt.setp(axdr6.get_yticklabels(), fontsize = 7)
axdr6.yaxis.set_label_position("right")
plt.grid(True)

plt.suptitle('Dutch roll', fontsize=16)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig('Graphs/dutch_roll.png', dpi = 300)
plt.show()

#Roll damping
rd_initYA = [0]
rd_initRA = [roll_damping['Roll_angle'].iloc[0]]
rd_initRR = [roll_damping['bRollRate'].iloc[0]]
rd_initYR = [roll_damping['bYawRate'].iloc[0]]

t6 = roll_damping['time']-roll_damping['time'].iloc[0]
h6 = roll_damping['alt']
v6 = roll_damping['V_true']
rr6 = roll_damping['bRollRate']
yr6 = roll_damping['bYawRate']
ra6 = roll_damping['Roll_angle']
d_a6 = roll_damping['d_a']
d_r6 = roll_damping['d_r']

axrd1 = plt.subplot(321)
plt.plot(t6, h6)
plt.ylabel('Altitude [ft]', fontsize = 5)
plt.setp(axrd1.get_xticklabels(), visible = False)
plt.setp(axrd1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axrd2 = plt.subplot(322)
plt.plot(t6, ra6)
plt.ylabel('Roll angle [deg]', fontsize = 5)
plt.setp(axrd2.get_xticklabels(), visible = False)
axrd2.yaxis.tick_right()
plt.setp(axrd2.get_yticklabels(), fontsize = 7)
axrd2.yaxis.set_label_position("right")
plt.grid(True)

axrd3 = plt.subplot(323, sharex=axrd1)
plt.plot(t6, v6)
plt.ylabel('True airspeed [V]', fontsize = 5)
plt.setp(axrd3.get_xticklabels(), visible = False)
plt.setp(axrd3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axrd4 = plt.subplot(324, sharex=axrd2)
plt.plot(t6, rr6)
plt.ylabel('Roll rate [deg/sec]', fontsize = 5)
plt.setp(axrd4.get_xticklabels(), visible = False)
axrd4.yaxis.tick_right()
plt.setp(axrd4.get_yticklabels(), fontsize = 7)
axrd4.yaxis.set_label_position("right")
plt.grid(True)

axrd5 = plt.subplot(325, sharex=axrd1)
plt.plot(t6, d_a6, label = 'Aileron deflection')
plt.plot(t6, d_r6, label = 'Rudder deflection')
plt.ylabel('Pilot input', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axrd5.get_xticklabels(), fontsize = 7)
plt.setp(axrd5.get_yticklabels(), fontsize = 7)
plt.legend(loc = 'best', prop={'size': 5})
plt.grid(True)

axrd6 = plt.subplot(326, sharex=axrd2)
plt.plot(t6, yr6)
plt.ylabel('Yaw rate [deg/sec]', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axrd6.get_xticklabels(), fontsize = 7)
axrd6.yaxis.tick_right()
plt.setp(axrd6.get_yticklabels(), fontsize = 7)
axrd6.yaxis.set_label_position("right")
plt.grid(True)

plt.suptitle('A-periodic roll', fontsize=16)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig('Graphs/aperiodic_roll.png', dpi = 300)
plt.show()

initvalues = [sp1_initV + sp1_initAOA + sp1_initPA + sp1_initPR,
              sp2_initV + sp2_initAOA + sp2_initPA + sp2_initPR,
              ph_initV  + ph_initAOA  + ph_initPA  + ph_initPR,
              spiral_initYA + spiral_initRA + spiral_initRR + spiral_initYR,
              dr_initYA + dr_initRA + dr_initRR + dr_initYR,
              rd_initYA + rd_initRA + rd_initRR + rd_initYR]

df_init = pd.DataFrame(initvalues, columns = ['V_true/Yaw_angle', 'Angle_of_attack/Roll_angle', 'Pitch_angle/Roll_rate', 'Pitch_rate/Yaw_rate'])
sp1_input = pd.concat([t1, d_e1], axis=1)
sp2_input = pd.concat([t2, d_e2], axis=1)
ph_input = pd.concat([t3, d_e3], axis=1)
spiral_input = pd.concat([t4, d_a4, d_r4], axis=1)
dr_input = pd.concat([t5, d_a5, d_r5], axis=1)
rd_input = pd.concat([t6, d_a6, d_r6], axis=1)

df_init.to_csv(r'input_data/initial_conditions.txt', header=None, index=None, sep=':', mode='a')
sp1_input.to_csv(r'input_data/input_short_period1.txt', header=None, index=None, sep=':', mode='a')
sp2_input.to_csv(r'input_data/input_short_period2.txt', header=None, index=None, sep=':', mode='a')
ph_input.to_csv(r'input_data/input_phugoid.txt', header=None, index=None, sep=':', mode='a')
spiral_input.to_csv(r'input_data/input_spiral.txt', header=None, index=None, sep=':', mode='a')
dr_input.to_csv(r'input_data/input_dutch_roll.txt', header=None, index=None, sep=':', mode='a')
rd_input.to_csv(r'input_data/input_aperiodic_roll.txt', header=None, index=None, sep=':', mode='a')

'''
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

axs8 = axs6[1].twinx()
axs8.scatter(x5,b5, s=0.6, c='red')
axs8.set_ylabel('Rudder deflection [deg]')
axs8.tick_params(axis='y', labelcolor='red')
plt.show()
'''

#pd.DataFrame.to_excel(short_period, 'short_period.xlsx')
#pd.DataFrame.to_excel(short_period2, 'short_period2.xlsx')
#pd.DataFrame.to_excel(phugoid, 'phugoid.xlsx')
#pd.DataFrame.to_excel(spiral, 'spiral.xlsx')
#pd.DataFrame.to_excel(dutch_roll, 'dutch_roll.xlsx')
#pd.DataFrame.to_excel(roll_damping, 'roll_damping.xlsx')


print('done')