from math import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from control import *

short_period = pd.read_excel('type_of_motion/short_period.xlsx')
short_period2 = pd.read_excel('type_of_motion/phugoid.xlsx')
phugoid = pd.read_excel('type_of_motion/phugoid.xlsx')
spiral = pd.read_excel('type_of_motion/dutch_roll.xlsx')
dutch_roll = pd.read_excel('type_of_motion/aperiodic_roll.xlsx')
roll_damping = pd.read_excel('type_of_motion/aperiodic_roll.xlsx')

short_period.drop(short_period.head(5970).index,inplace=True)
short_period.drop(short_period.tail(4541).index,inplace=True)

short_period2.drop(short_period2.head(470).index,inplace=True)
short_period2.drop(short_period2.tail(1805).index,inplace=True)

phugoid.drop(phugoid.head(250).index,inplace=True)
phugoid.drop(phugoid.tail(300).index,inplace=True)

spiral.drop(spiral.head(30).index,inplace=True)
spiral.drop(spiral.tail(470).index,inplace=True)

dutch_roll.drop(dutch_roll.head(110).index,inplace=True)
dutch_roll.drop(dutch_roll.tail(520).index,inplace=True)

roll_damping.drop(roll_damping.head(704).index,inplace=True)
roll_damping.drop(roll_damping.tail(0).index,inplace=True)

rho0 = 1.225
lmbda = -0.0065
g = 9.81
R = 287.05
Temp0 = 288.15
mass = 6152

def symmetric(Data, filename):
    
    file = 'Plots/' + filename + '.png'
    
    Vt0 = Data['V_true'].iloc[0]*0.514444444
    alpha0 = Data['AOA'].iloc[0]*(pi/180)
    th0 = Data['Pitch_angle'].iloc[0]*(pi/180)
    hp0 = Data['alt'].iloc[0]*0.3048
    time = Data['time']-Data['time'].iloc[0]
    d_e = (Data['d_e']-Data['d_e'].iloc[0])*(pi/180)
    m = mass-(Data['fuel_used_l'].iloc[0]+Data['fuel_used_r'].iloc[0])*0.45359237
    
    rho    = rho0 * ((1+(lmbda * hp0 / Temp0)))**(-((g / (lmbda*R)) + 1))
    
    S      = 30
    c      = 2.0569
    b      = 15.911
    
    Cd0    = 0.04
    Cla    = 5.084
    e      = 0.8
    
    Kxx2   = 0.019
    Kyy2   = 1.3925
    Kzz2   = 0.042
    Kxz    = 0.002
    
    Cxu    = -0.0279
    Cxa    = -0.4797
    Cxadot = 0.0833
    Cxq    = -0.2817
    Cxd    = -0.0373
    
    Czu    = -0.3762
    Cza    = -5.7434
    Czadot = -0.0035
    Czq    = -5.6629
    Czd    = -0.6961
    
    Cm0    = 0.0297
    Cmu    = 0.0699
    Cma    = -0.5626
    Cmadot = 0.1780
    Cmq    = -8.7941
    Cmd    = -1.1642
    
    W      = m*g
    muc    = m/(rho*S*c)
    Cx0    = (W*sin(th0))/(0.5*rho*(Vt0**2)*S)
    Cz0    = -(W*cos(th0))/(0.5*rho*(Vt0**2)*S)
    
    C1 = np.asarray([[-2*muc*(c/Vt0)     , 0                     , 0     , 0                     ], 
                     [0                  , (Czadot-2*muc)*(c/Vt0), 0     , 0                     ], 
                     [0                  , 0                     , -c/Vt0, 0                     ], 
                     [0                  , Cmadot*(c/Vt0)        , 0     , -2*muc*Kyy2*(c/Vt0)   ]])

    C2 = np.asarray([[Cxu, Cxa, Cz0 , Cxq        ], 
                     [Czu, Cza, -Cx0, (Czq+2*muc)], 
                     [0  , 0  , 0   , 1          ], 
                     [Cmu, Cma, 0   , Cmq        ]])
    
    C3 = np.asarray([[-Cxd], 
                     [-Czd], 
                     [0   ], 
                     [-Cmd]])
    
    A = np.multiply(-1, np.matmul(np.linalg.inv(C1), C2))
    B = np.matmul(np.linalg.inv(C1), C3)
    C = np.asarray([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0, 0.0], 
                    [0.0, 0.0, 0.0, 1.0]])
    D = 0.0
    
    duration = time.iloc[-1]
    t = np.asarray(time)
    f = np.asarray(d_e)
    
    sym = StateSpace(A, B, C, D)
    t, y, x = forced_response(sym, t, f)
    
    y_v = Vt0*(y[0]+1)
    y_a = y[1]*(180/pi)+alpha0*(180/pi)
    y_pa = y[2]*(180/pi)+th0*(180/pi)
    y_pr = y[3]/(c/Vt0)*(180/pi)+Data['bPitchRate'].iloc[0]
    '''
    ax1 = plt.subplot(221)
    plt.plot(t, y_v)
    plt.title('Velocity', fontsize = 10)
    plt.ylabel('True Airspeed [m/s]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    plt.setp(ax1.get_xticklabels(), fontsize = 7)
    plt.setp(ax1.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(0,duration)
    
    ax2 = plt.subplot(222)
    plt.plot(t, y_a)
    plt.title('Angle of Attack', fontsize = 10)
    plt.ylabel('Angle of Attack [deg]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.setp(ax2.get_xticklabels(), fontsize = 7)
    plt.setp(ax2.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(0,duration)
    
    ax3 = plt.subplot(223)
    plt.plot(t, y_pa)
    plt.title('Pitch Angle', fontsize = 10)
    plt.ylabel('Pitch Angle [deg]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    plt.setp(ax3.get_xticklabels(), fontsize = 7)
    plt.setp(ax3.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(0,duration)
    
    ax4 = plt.subplot(224)
    plt.plot(t, y_pr)
    plt.title('Pitch Rate', fontsize = 10)
    plt.ylabel('Pitch Rate [deg/sec]', fontsize = 7)
    plt.xlabel('time [sec]', fontsize = 7)
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    plt.setp(ax4.get_xticklabels(), fontsize = 7)
    plt.setp(ax4.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(0,duration)
    
    plt.suptitle('Symmetric response', fontsize=16)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.savefig(file, dpi = 600)
    plt.show()
    '''
    return t, y_v, y_a, y_pa, y_pr

def asymmetric(Data, filename):
        
    file = 'Plots/' + filename + '.png'
    
    Vt0 = Data['V_true'].iloc[0]*0.514444444
    alpha0 = Data['AOA'].iloc[0]*(pi/180)
    th0 = Data['Pitch_angle'].iloc[0]*(pi/180)
    hp0 = Data['alt'].iloc[0]*0.3048
    time = Data['time']-Data['time'].iloc[0]
    d_a = (Data['d_a']-Data['d_a'].iloc[0])*(pi/180)
    d_r = (Data['d_r']-Data['d_r'].iloc[0])*(pi/180)
    m = mass-(Data['fuel_used_l'].iloc[0]+Data['fuel_used_r'].iloc[0])*0.45359237
    
    rho    = rho0 * ((1+(lmbda * hp0 / Temp0)))**(-((g / (lmbda*R)) + 1))
    
    S      = 30
    c      = 2.0569
    b      = 15.911
    
    Cd0    = 0.04
    Cla    = 5.084
    e      = 0.8
    
    Kxx2   = 0.019
    Kyy2   = 1.3925
    Kzz2   = 0.042
    Kxz    = 0.002
    
    Cyb    = -0.7500
    Cybdot = 0
    Cyp    = -0.0304
    Cyr    = 0.8495
    Cyda   = -0.0400
    Cydr   = 0.2300
    
    Clb    = -0.1026
    Clp    = -0.7108
    Clr    = 0.2376
    Clda   = -0.2309
    Cldr   = 0.0344
    
    Cnb    = 0.1348
    Cnbdot = 0
    Cnp    = -0.0602
    Cnr    = -0.2061
    Cnda   = -0.0120
    Cndr   = -0.0939
    
    W      = m*9.81
    muc    = m/(rho*S*c)
    mub    = m/(rho*S*b)
    
    Cx0    = (W*sin(th0))/(0.5*rho*(Vt0**2)*S)
    Cz0    = -(W*cos(th0))/(0.5*rho*(Vt0**2)*S)
    CL     = W/(0.5*rho*(Vt0**2)*S)
    
    C1 = np.asarray([[(Cybdot-2*mub)*(b/Vt0), 0           , 0                  , 0                  ],
                     [0                     , -0.5*(b/Vt0), 0                  , 0                  ],
                     [0                     , 0           , -4*mub*Kxx2*(b/Vt0), 4*mub*Kxz*(b/Vt0)  ],
                     [Cnbdot*(b/Vt0)        , 0           , 4*mub*Kxz*(b/Vt0)  , -4*mub*Kzz2*(b/Vt0)]])
    
    C2 = np.asarray([[Cyb, CL, Cyp, Cyr-4*mub],
                     [0  , 0 , 1  , 0        ],
                     [Clb, 0 , Clp, Clr      ],
                     [Cnb, 0 , Cnp, Cnr      ]])
    
    C3 = np.asarray([[-Cyda, -Cydr],
                     [0    , 0    ],
                     [-Clda, -Cldr],
                     [-Cnda, -Cndr]])
    
    A = np.multiply(-1, np.matmul(np.linalg.inv(C1), C2))
    B = np.multiply(-1, np.matmul(np.linalg.inv(C1), C3))
    #B = np.matmul(np.linalg.inv(C1), C3)
    C = np.asarray([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0], 
                     [0.0, 0.0, 1.0, 0.0], 
                     [0.0, 0.0, 0.0, 1.0]])
    D = 0.0
    
    duration = time.iloc[-1]
    t = np.asarray(time)
    f = np.asarray([d_a, d_r])
    
    asym = StateSpace(A, B, C, D)
    t, y, x = forced_response(asym, t, f)
    
    y_ya = y[0]*(180/pi)
    y_ra = y[1]*(180/pi)+Data['Roll_angle'].iloc[0]
    y_rr = y[2]/(b/(2*Vt0))*(180/pi)+Data['bRollRate'].iloc[0]
    y_yr = y[3]/(b/(2*Vt0))*(180/pi)+Data['bYawRate'].iloc[0]
    '''
    axs1 = plt.subplot(221)
    plt.plot(t, y_ya)
    plt.title('Yaw angle', fontsize = 10)
    plt.ylabel('Yaw angle [rad]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    plt.setp(axs1.get_xticklabels(), fontsize = 7)
    plt.setp(axs1.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(0,duration)
    
    axs2 = plt.subplot(222)
    plt.plot(t, y_ra)
    plt.title('Roll Angle', fontsize = 10)
    plt.ylabel('Roll Angle [rad]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    axs2.yaxis.tick_right()
    axs2.yaxis.set_label_position("right")
    plt.setp(axs2.get_xticklabels(), fontsize = 7)
    plt.setp(axs2.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(0,duration)
    
    axs3 = plt.subplot(223)
    plt.plot(t, y_rr)
    plt.title('Roll Rate', fontsize = 10)
    plt.ylabel('Roll Rate [rad/sec]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    plt.setp(axs3.get_xticklabels(), fontsize = 7)
    plt.setp(axs3.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(0,duration)
    
    axs4 = plt.subplot(224)
    plt.plot(t, y_yr)
    plt.title('Yaw Rate', fontsize = 10)
    plt.ylabel('Yaw Rate [rad/sec]', fontsize = 7)
    plt.xlabel('time [sec]', fontsize = 7)
    axs4.yaxis.tick_right()
    axs4.yaxis.set_label_position("right")
    plt.setp(axs4.get_xticklabels(), fontsize = 7)
    plt.setp(axs4.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(0,duration)
    
    plt.suptitle('Symmetric response', fontsize=16)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    #plt.savefig(file, dpi = 600)
    plt.show()
    '''
    return t, y_ya, y_ra, y_rr, y_yr

#Short Period 1
sp1_initV = [short_period['V_true'].iloc[0]]
sp1_initAOA = [short_period['AOA'].iloc[0]]
sp1_initPA = [short_period['Pitch_angle'].iloc[0]]
sp1_initPR = [short_period['bPitchRate'].iloc[0]]

t1 = short_period['time']-short_period['time'].iloc[0]
h1 = short_period['alt']*0.3048
v1 = short_period['V_true']*0.5144444
aoa1 = short_period['AOA']
pr1 = short_period['bPitchRate']
pa1 = short_period['Pitch_angle']
d_e1 = short_period['d_e']
Det1 = short_period['Det']

sp1_t_num, sp1_v_num, sp1_aoa_num, sp1_pa_num, sp1_pr_num = symmetric(short_period, 'nothing')

axsp1 = plt.subplot(321)
plt.plot(t1, h1)
plt.ylabel('Altitude [m]', fontsize = 5)
plt.setp(axsp1.get_xticklabels(), visible = False)
plt.setp(axsp1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axsp2 = plt.subplot(322)
plt.plot(t1, pa1, label = 'Flight Data')
plt.plot(t1, sp1_pa_num, label = 'Numerical Model')
plt.ylabel('Pitch angle [deg]', fontsize = 5)
plt.setp(axsp2.get_xticklabels(), visible = False)
axsp2.yaxis.tick_right()
plt.setp(axsp2.get_yticklabels(), fontsize = 7)
axsp2.yaxis.set_label_position("right")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0., prop={'size': 7})
plt.grid(True)

axsp3 = plt.subplot(323, sharex=axsp1)
plt.plot(t1, v1)
plt.plot(t1, sp1_v_num)
plt.ylabel('True airspeed [m/s]', fontsize = 5)
plt.setp(axsp3.get_xticklabels(), visible = False)
plt.setp(axsp3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axsp4 = plt.subplot(324, sharex=axsp2)
plt.plot(t1, pr1)
plt.plot(t1, sp1_pr_num)
plt.ylabel('Pitch rate [deg/sec]', fontsize = 5)
plt.setp(axsp4.get_xticklabels(), visible = False)
axsp4.yaxis.tick_right()
plt.setp(axsp4.get_yticklabels(), fontsize = 7)
axsp4.yaxis.set_label_position("right")
plt.grid(True)

axsp5 = plt.subplot(325, sharex=axsp1)
plt.plot(t1, aoa1)
plt.plot(t1, sp1_aoa_num)
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
h2 = short_period2['alt']*0.3048
v2 = short_period2['V_true']*0.5144444
aoa2 = short_period2['AOA']
pr2 = short_period2['bPitchRate']
pa2 = short_period2['Pitch_angle']
d_e2 = short_period2['d_e']
Det2 = short_period2['Det']

sp2_t_num, sp2_v_num, sp2_aoa_num, sp2_pa_num, sp2_pr_num = symmetric(short_period2, '-')

axs1 = plt.subplot(321)
plt.plot(t2, h2)
plt.ylabel('Altitude [m]', fontsize = 5)
plt.setp(axs1.get_xticklabels(), visible = False)
plt.setp(axs1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axs2 = plt.subplot(322)
plt.plot(t2, pa2, label = 'Flight Data')
plt.plot(t2, sp2_pa_num, label = 'Numerical Model')
plt.ylabel('Pitch angle [deg]', fontsize = 5)
plt.setp(axs2.get_xticklabels(), visible = False)
axs2.yaxis.tick_right()
plt.setp(axs2.get_yticklabels(), fontsize = 7)
axs2.yaxis.set_label_position("right")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0., prop={'size': 7})
plt.grid(True)

axs3 = plt.subplot(323, sharex=axs1)
plt.plot(t2, v2)
plt.plot(t2, sp2_v_num)
plt.ylabel('True airspeed [m/s]', fontsize = 5)
plt.setp(axs3.get_xticklabels(), visible = False)
plt.setp(axs3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axs4 = plt.subplot(324, sharex=axs2)
plt.plot(t2, pr2)
plt.plot(t2, sp2_pr_num)
plt.ylabel('Pitch rate [deg/sec]', fontsize = 5)
plt.setp(axs4.get_xticklabels(), visible = False)
axs4.yaxis.tick_right()
plt.setp(axs4.get_yticklabels(), fontsize = 7)
axs4.yaxis.set_label_position("right")
plt.grid(True)

axs5 = plt.subplot(325, sharex=axs1)
plt.plot(t2, aoa2)
plt.plot(t2, sp2_aoa_num)
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
h3 = phugoid['alt']*0.3048
v3 = phugoid['V_true']*0.5144444
aoa3 = phugoid['AOA']
pr3 = phugoid['bPitchRate']
pa3 = phugoid['Pitch_angle']
d_e3 = phugoid['d_e']
Det3 = phugoid['Det']

ph_t_num, ph_v_num, ph_aoa_num, ph_pa_num, ph_pr_num = symmetric(phugoid, '-')

axph1 = plt.subplot(321)
plt.plot(t3, h3)
plt.ylabel('Altitude [m]', fontsize = 5)
plt.setp(axph1.get_xticklabels(), visible = False)
plt.setp(axph1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axph2 = plt.subplot(322)
plt.plot(t3, pa3, label = 'Flight Data')
plt.plot(t3, ph_pa_num, label = 'Numerical Model')
plt.ylabel('Pitch angle [deg]', fontsize = 5)
plt.setp(axph2.get_xticklabels(), visible = False)
axph2.yaxis.tick_right()
plt.setp(axph2.get_yticklabels(), fontsize = 7)
axph2.yaxis.set_label_position("right")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0., prop={'size': 7})
plt.grid(True)

axph3 = plt.subplot(323, sharex=axph1)
plt.plot(t3, v3)
plt.plot(t3, ph_v_num)
plt.ylabel('True airspeed [m/s]', fontsize = 5)
plt.setp(axph3.get_xticklabels(), visible = False)
plt.setp(axph3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axph4 = plt.subplot(324, sharex=axph2)
plt.plot(t3, pr3)
plt.plot(t3, ph_pr_num)
plt.ylabel('Pitch rate [deg/sec]', fontsize = 5)
plt.setp(axph4.get_xticklabels(), visible = False)
axph4.yaxis.tick_right()
plt.setp(axph4.get_yticklabels(), fontsize = 7)
axph4.yaxis.set_label_position("right")
plt.grid(True)

axph5 = plt.subplot(325, sharex=axph1)
plt.plot(t3, aoa3)
plt.plot(t3, ph_aoa_num)
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
h4 = spiral['alt']*0.3048
v4 = spiral['V_true']*0.5144444
rr4 = spiral['bRollRate']
yr4 = spiral['bYawRate']
ra4 = spiral['Roll_angle']
d_a4 = spiral['d_a']
d_r4 = spiral['d_r']

spr_t_num, spr_ya_num, spr_ra_num, spr_rr_num, spr_yr_num = asymmetric(spiral, '-')

axsr1 = plt.subplot(321)
plt.plot(t4, h4)
plt.ylabel('Altitude [m]', fontsize = 5)
plt.setp(axsr1.get_xticklabels(), visible = False)
plt.setp(axsr1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axsr2 = plt.subplot(322)
plt.plot(t4, ra4, label = 'Flight Data')
plt.plot(t4, spr_ra_num, label = 'Numerical Model')
plt.ylabel('Roll angle [deg]', fontsize = 5)
plt.setp(axsr2.get_xticklabels(), visible = False)
axsr2.yaxis.tick_right()
plt.setp(axsr2.get_yticklabels(), fontsize = 7)
axsr2.yaxis.set_label_position("right")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0., prop={'size': 7})
plt.grid(True)

axsr3 = plt.subplot(323, sharex=axsr1)
plt.plot(t4, v4)
plt.ylabel('True airspeed [m/s]', fontsize = 5)
plt.setp(axsr3.get_xticklabels(), visible = False)
plt.setp(axsr3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axsr4 = plt.subplot(324, sharex=axsr2)
plt.plot(t4, rr4)
plt.plot(t4, spr_rr_num)
plt.ylabel('Roll rate [deg/sec]', fontsize = 5)
plt.setp(axsr4.get_xticklabels(), visible = False)
axsr4.yaxis.tick_right()
plt.setp(axsr4.get_yticklabels(), fontsize = 7)
axsr4.yaxis.set_label_position("right")
plt.grid(True)

axsr5 = plt.subplot(325, sharex=axsr1)
plt.plot(t4, d_a4,c = 'r', label = 'Aileron deflection $\u03B4_a$')
plt.plot(t4, d_r4,c = 'g', label = 'Rudder deflection $\u03B4_r$')
plt.ylabel('Pilot input', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axsr5.get_xticklabels(), fontsize = 7)
plt.setp(axsr5.get_yticklabels(), fontsize = 7)
plt.legend(loc = 'best', prop={'size': 4})
plt.grid(True)

axsr6 = plt.subplot(326, sharex=axsr2)
plt.plot(t4, yr4)
plt.plot(t4, spr_yr_num)
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
h5 = dutch_roll['alt']*0.3048
v5 = dutch_roll['V_true']*0.5144444
rr5 = dutch_roll['bRollRate']
yr5 = dutch_roll['bYawRate']
ra5 = dutch_roll['Roll_angle']
d_a5 = dutch_roll['d_a']
d_r5 = dutch_roll['d_r']

dr_t_num, dr_ya_num, dr_ra_num, dr_rr_num, dr_yr_num = asymmetric(dutch_roll, 'hello')

axdr1 = plt.subplot(321)
plt.plot(t5, h5)
plt.ylabel('Altitude [m]', fontsize = 5)
plt.setp(axdr1.get_xticklabels(), visible = False)
plt.setp(axdr1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axdr2 = plt.subplot(322)
plt.plot(t5, ra5, label = 'Flight Data')
plt.plot(t5, dr_ra_num, label = 'Numerical Model')
plt.ylabel('Roll angle [deg]', fontsize = 5)
plt.setp(axdr2.get_xticklabels(), visible = False)
axdr2.yaxis.tick_right()
plt.setp(axdr2.get_yticklabels(), fontsize = 7)
axdr2.yaxis.set_label_position("right")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0., prop={'size': 7})
plt.grid(True)

axdr3 = plt.subplot(323, sharex=axdr1)
plt.plot(t5, v5)
plt.ylabel('True airspeed [m/s]', fontsize = 5)
plt.setp(axdr3.get_xticklabels(), visible = False)
plt.setp(axdr3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axdr4 = plt.subplot(324, sharex=axdr2)
plt.plot(t5, rr5)
plt.plot(t5, dr_rr_num)
plt.ylabel('Roll rate [deg/sec]', fontsize = 5)
plt.setp(axdr4.get_xticklabels(), visible = False)
axdr4.yaxis.tick_right()
plt.setp(axdr4.get_yticklabels(), fontsize = 7)
axdr4.yaxis.set_label_position("right")
plt.grid(True)

axdr5 = plt.subplot(325, sharex=axdr1)
plt.plot(t5, d_a5,c = 'r', label = 'Aileron deflection $\u03B4_a$')
plt.plot(t5, d_r5,c = 'g', label = 'Rudder deflection $\u03B4_r$')
plt.ylabel('Pilot input', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axdr5.get_xticklabels(), fontsize = 7)
plt.setp(axdr5.get_yticklabels(), fontsize = 7)
plt.legend(loc = 'best', prop={'size': 4})
plt.grid(True)

axdr6 = plt.subplot(326, sharex=axdr2)
plt.plot(t5, yr5)
plt.plot(t5, dr_yr_num)
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
h6 = roll_damping['alt']*0.3048
v6 = roll_damping['V_true']*0.5144444
rr6 = roll_damping['bRollRate']
yr6 = roll_damping['bYawRate']
ra6 = roll_damping['Roll_angle']
d_a6 = roll_damping['d_a']
d_r6 = roll_damping['d_r']

rd_t_num, rd_ya_num, rd_ra_num, rd_rr_num, rd_yr_num = asymmetric(roll_damping, 'hello')

axrd1 = plt.subplot(321)
plt.plot(t6, h6)
plt.ylabel('Altitude [m]', fontsize = 5)
plt.setp(axrd1.get_xticklabels(), visible = False)
plt.setp(axrd1.get_yticklabels(), fontsize = 7)
plt.grid(True)

axrd2 = plt.subplot(322)
plt.plot(t6, ra6, label = 'Flight Data')
plt.plot(t6, rd_ra_num, label = 'Numerical Model')
plt.ylabel('Roll angle [deg]', fontsize = 5)
plt.setp(axrd2.get_xticklabels(), visible = False)
axrd2.yaxis.tick_right()
plt.setp(axrd2.get_yticklabels(), fontsize = 7)
axrd2.yaxis.set_label_position("right")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0., prop={'size': 7})
plt.grid(True)

axrd3 = plt.subplot(323, sharex=axrd1)
plt.plot(t6, v6)
plt.ylabel('True airspeed m/s]', fontsize = 5)
plt.setp(axrd3.get_xticklabels(), visible = False)
plt.setp(axrd3.get_yticklabels(), fontsize = 7)
plt.grid(True)

axrd4 = plt.subplot(324, sharex=axrd2)
plt.plot(t6, rr6)
plt.plot(t6, rd_rr_num)
plt.ylabel('Roll rate [deg/sec]', fontsize = 5)
plt.setp(axrd4.get_xticklabels(), visible = False)
axrd4.yaxis.tick_right()
plt.setp(axrd4.get_yticklabels(), fontsize = 7)
axrd4.yaxis.set_label_position("right")
plt.grid(True)

axrd5 = plt.subplot(325, sharex=axrd1)
plt.plot(t6, d_a6,c = 'r', label = 'Aileron deflection $\u03B4_a$')
plt.plot(t6, d_r6,c = 'g', label = 'Rudder deflection $\u03B4_e')
plt.ylabel('Pilot input', fontsize = 5)
plt.xlabel('Elapsed time [sec]', fontsize = 5)
plt.setp(axrd5.get_xticklabels(), fontsize = 7)
plt.setp(axrd5.get_yticklabels(), fontsize = 7)
axrd5.legend(loc = 'best', prop={'size': 4})
plt.grid(True)

axrd6 = plt.subplot(326, sharex=axrd2)
plt.plot(t6, yr6)
plt.plot(t6, rd_yr_num)
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

'''
df_init.to_csv(r'input_data/initial_conditions.txt', header=None, index=None, sep=':', mode='a')
sp1_input.to_csv(r'input_data/input_short_period1.txt', header=None, index=None, sep=':', mode='a')
sp2_input.to_csv(r'input_data/input_short_period2.txt', header=None, index=None, sep=':', mode='a')
ph_input.to_csv(r'input_data/input_phugoid.txt', header=None, index=None, sep=':', mode='a')
spiral_input.to_csv(r'input_data/input_spiral.txt', header=None, index=None, sep=':', mode='a')
dr_input.to_csv(r'input_data/input_dutch_roll.txt', header=None, index=None, sep=':', mode='a')
rd_input.to_csv(r'input_data/input_aperiodic_roll.txt', header=None, index=None, sep=':', mode='a')
'''
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

pd.DataFrame.to_excel(short_period, 'short_period.xlsx')
pd.DataFrame.to_excel(short_period2, 'short_period2.xlsx')
pd.DataFrame.to_excel(phugoid, 'phugoid.xlsx')
pd.DataFrame.to_excel(spiral, 'spiral.xlsx')
pd.DataFrame.to_excel(dutch_roll, 'dutch_roll.xlsx')
pd.DataFrame.to_excel(roll_damping, 'roll_damping.xlsx')


print('done')