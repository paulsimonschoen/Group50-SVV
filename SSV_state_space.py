from math import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from control import *

rho0 = 1.225
lmbda = -0.0065
g = 9.81
R = 287.05
Temp0 = 288.15
mass = 6152

def symmetric(time, X0, filename, colour):
    
    file = 'reading flight data/Graphs/' + filename + '.png'
    
    Vt0 = 160*0.514444444
    alpha0 = 6*(pi/180)
    th0 = 5*(pi/180)
    hp0 = 7421.84*0.3048
    m = 6000
    
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
    X0 = [X0[0], X0[1], X0[2], X0[3]*(c/Vt0)]
    t = np.linspace(0, time, time*10)
    
    sym = StateSpace(A, B, C, D)
    t, y = initial_response(sym, t, X0)
    margin = 0.05
    
    y_v = Vt0*(y[0]+1)
    y_a = y[1]*(180/pi)
    y_pa = y[2]*(180/pi)
    y_pr = y[3]/(c/Vt0)*(180/pi)
    
    ax1 = plt.subplot(221)
    plt.plot(t, y_v, c = colour)
    plt.title('Velocity', fontsize = 10)
    plt.ylabel('True Airspeed [m/s]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    plt.setp(ax1.get_xticklabels(), fontsize = 7)
    plt.setp(ax1.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(-time*margin,time*(1+margin))
    
    ax2 = plt.subplot(222)
    plt.plot(t, y_a, c = colour)
    plt.title('Angle of Attack', fontsize = 10)
    plt.ylabel('Angle of Attack [deg]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.setp(ax2.get_xticklabels(), fontsize = 7)
    plt.setp(ax2.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(-time*margin,time*(1+margin))
    
    ax3 = plt.subplot(223)
    plt.plot(t, y_pa, c = colour)
    plt.title('Pitch Angle', fontsize = 10)
    plt.ylabel('Pitch Angle [deg]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    plt.setp(ax3.get_xticklabels(), fontsize = 7)
    plt.setp(ax3.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(-time*margin,time*(1+margin))
    
    ax4 = plt.subplot(224)
    plt.plot(t, y_pr, c = colour)
    plt.title('Pitch Rate', fontsize = 10)
    plt.ylabel('Pitch Rate [deg/sec]', fontsize = 7)
    plt.xlabel('time [sec]', fontsize = 7)
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    plt.setp(ax4.get_xticklabels(), fontsize = 7)
    plt.setp(ax4.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(-time*margin,time*(1+margin))
    
    plt.suptitle(filename, fontsize=16)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.savefig(file, dpi = 600)
    plt.show()
    
    print(damp(sym))
    
    return t, y_v, y_a, y_pa, y_pr

def asymmetric(time, X0, filename, colour):
        
    file = 'reading flight data/Graphs/' + filename + '.png'
    
    Vt0 = 160*0.514444444
    alpha0 = 6*(pi/180)
    th0 = 5*(pi/180)
    hp0 = 7421.84*0.3048
    m = 6000
    
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
    C = np.asarray([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0], 
                     [0.0, 0.0, 1.0, 0.0], 
                     [0.0, 0.0, 0.0, 1.0]])
    D = 0.0
    X0 = [X0[0], X0[1], X0[2]*(b/(2*Vt0)), X0[3]*(b/(2*Vt0))]
    t = np.linspace(0, time, time*10)
    margin = 0.05
    
    asym = StateSpace(A, B, C, D)
    t, y = initial_response(asym, t, X0)
    
    y_ya = y[0]*(180/pi)
    y_ra = y[1]*(180/pi)
    y_rr = y[2]/(b/(2*Vt0))*(180/pi)
    y_yr = y[3]/(b/(2*Vt0))*(180/pi)
    
    axs1 = plt.subplot(221)
    plt.plot(t, y_ya, c = colour)
    plt.title('Yaw Angle', fontsize = 10)
    plt.ylabel('Yaw Angle [deg]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    plt.setp(axs1.get_xticklabels(), fontsize = 7)
    plt.setp(axs1.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(-time*margin,time*(1+margin))
    
    axs2 = plt.subplot(222)
    plt.plot(t, y_ra, c = colour)
    plt.title('Roll Angle', fontsize = 10)
    plt.ylabel('Roll Angle [deg]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    axs2.yaxis.tick_right()
    axs2.yaxis.set_label_position("right")
    plt.setp(axs2.get_xticklabels(), fontsize = 7)
    plt.setp(axs2.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(-time*margin,time*(1+margin))
    
    axs3 = plt.subplot(223)
    plt.plot(t, y_rr, c = colour)
    plt.title('Roll Rate', fontsize = 10)
    plt.ylabel('Roll Rate [deg/sec]', fontsize = 7)
    plt.xlabel('Time [sec]', fontsize = 7)
    plt.setp(axs3.get_xticklabels(), fontsize = 7)
    plt.setp(axs3.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(-time*margin,time*(1+margin))
    
    axs4 = plt.subplot(224)
    plt.plot(t, y_yr, c = colour)
    plt.title('Yaw Rate', fontsize = 10)
    plt.ylabel('Yaw Rate [deg/sec]', fontsize = 7)
    plt.xlabel('time [sec]', fontsize = 7)
    axs4.yaxis.tick_right()
    axs4.yaxis.set_label_position("right")
    plt.setp(axs4.get_xticklabels(), fontsize = 7)
    plt.setp(axs4.get_yticklabels(), fontsize = 7)
    plt.grid(True)
    plt.xlim(-time*margin,time*(1+margin))
    
    plt.suptitle(filename, fontsize=16)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.savefig(file, dpi = 600)
    plt.show()
    
    print(damp(asym))
    
    return t, y_ya, y_ra, y_rr, y_yr

symmetric(100, [0.1, 0, 0, 0], 'Initial Velocity Disturbance', 'b')
symmetric(100, [0, 0.005, 0, 0], 'Initial Angle of Attack Disturbance', 'r')
symmetric(100, [0, 0, 0.09, 0], 'Initial Pitch Angle Disturbance', 'g')
symmetric(100, [0, 0, 0, 0.02], 'Initial Pitch Rate Disturbance', 'm')

asymmetric(15, [0.009, 0, 0, 0], 'Initial Yaw Angle Disturbance', 'b')
asymmetric(15, [0, 0.3, 0, 0], 'Initial Roll Angle Disturbance', 'r')
asymmetric(15, [0, 0, 0.2, 0], 'Initial Roll Rate Disturbance', 'g')
asymmetric(15, [0, 0, 0, 0.2], 'Initial Yaw Rate Disturbance', 'm')