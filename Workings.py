import numpy as np
from StateSpaceSystem import State_Space as ss
from math import sin, cos, pi

S    = 30.00	          # wing area [m^2]
m    = 5467.17068
Cxu  = -0.02792
Cxa  = +0.47966  # Positive! (has been erroneously negative since 1993)
Cxda = +0.08330
Cxq  = -0.28170
Cxde = -0.03728

Czu  = -0.37616
Cza  = -5.74340
Czda = -0.00350
Czq  = -5.66290
Czde = -0.69612

Cmu  = +0.06990
Cmda = +0.17800
Cmq  = -8.79415

Cyb  = -0.7500
Cydb = 0
Cyp  = -0.0304
Cyr  = +0.8495
Cyda = -0.0400
Cydr = +0.2300

Clb  = -0.10260
Clp  = -0.71085
Clr  = +0.23760
Clda = -0.23088
Cldr = +0.03440

Cnb  = +0.1348
Cndb = 0
Cnp  = -0.0602
Cnr  = -0.2061
Cnda = -0.0120
Cndr = -0.0939

c    = 2.0569
rho0 = 1.2250  # air density at sea level [kg/m^3]
lmbda= -0.0065  # temperature gradient in ISA [K/m]
Temp0= 288.15  # temperature at sea level in ISA [K]
R    = 287.05  # specific gas constant [m^2/sec^2K]
g    = 9.81
W    = m * g  # [N]       (aircraft weight)
Kxx  = 0.019
Kzz  = 0.042
Kxz  = 0.002
Kyy  = 1.25 * 1.114
Cma  = -0.04606522609004041  # longitudinal stabilty [ ]
Cmde = -0.1023671690889787
b    = 15.911  # wing span [m]


def GetAB(P, Q, R):
    A = np.linalg.inv(P) * Q
    B = np.linalg.inv(P) * R
    return A, B


def symmetric(type):


    if type is "SP1":
        t = 2091
        Temp0 = -7.75 + 273.15
        V0 = 198
        th0 = 0
        hp0 = 6816.36
        rho = rho0 * pow((1 + (lmbda * hp0 / Temp0)), (-((g / (lmbda * R)) + 1)))
        muc = m / (rho * S * c)
        Cx0 = W * sin(th0) / (0.5 * rho * V0 ** 2 * S)
        Cz0 = -W * cos(th0) / (0.5 * rho * V0 ** 2 * S)
        dc = c / V0
        u = [0.6]+[0]*(t-1)
        q0 = 0.5 * rho * V0 ** 2
        x0 = np.mat([[0], [0], [0], [q0]])

    elif type is "SP2":
        t = 2580
        Temp0 = -8.5 + 273.15
        V0 = 160
        th0 = 0
        hp0 = 7424
        rho = rho0 * pow(((1 + (lmbda * hp0 / Temp0))), (-((g / (lmbda * R)) + 1)))
        muc = m / (rho * S * c)
        Cx0 = W * sin(th0) / (0.5 * rho * V0 ** 2 * S)
        Cz0 = -W * cos(th0) / (0.5 * rho * V0 ** 2 * S)
        dc = c / V0
        u = [-1.25] + [-0.75] * (t-1)
        q0 = 0.5 * rho * V0 ** 2
        x0 = np.mat([[0], [0], [0], [q0]])

    else:
        t = 2575
        Temp0 = -8.5 + 273.15
        V0 = 160
        th0 = 0
        hp0 = 7422.69
        rho = rho0 * pow(((1 + (lmbda * hp0 / Temp0))), (-((g / (lmbda * R)) + 1)))
        muc = m / (rho * S * c)
        Cx0 = W * sin(th0) / (0.5 * rho * V0 ** 2 * S)
        Cz0 = -W * cos(th0) / (0.5 * rho * V0 ** 2 * S)
        dc = c / V0
        u = [-1.25] + [-0.75] * (t-1)
        q0 = 0.5 * rho * V0 ** 2
        x0 = np.mat([[0], [0], [0], [q0]])

    SP = np.mat([[-2 * muc * dc, 0, 0, 0], [0, (Czda - 2 * muc) * dc, 0, 0], [0, 0, -dc, 0],
                    [0, Cmda * dc, 0, -2 * muc * Kyy ** 2 * dc ** 2]])
    SQ = np.mat(
        [[-Cxu, -Cxa, -Cz0, -Cxq], [-Czu, -Cza, Cx0, -(Czq + 2 * muc)], [0, 0, 0, -1], [-Cmu, -Cma, 0, -Cmq]])
    SR = np.mat([[-Cxde], [-Czde], [0], [Cmde]])
    C = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    D = np.mat([[0], [0], [0], [0]])

    A, B = GetAB(SP, SQ, SR)

    stateSpace = ss(A, B, C, D, x0)
    eig,vecs = stateSpace.get_eig()
    eig = [val*dc for val in eig]
    #stateSpace.plot_resp(u)
    print(eig)


def Asymmetric(type):
    x0 = 1

    if type is "spiral":
        t = 2870
        Temp0 = -9 + 273.15
        V0 = 162
        hp0 = 7827.27
        rho = rho0 * pow(((1 + (lmbda * hp0 / Temp0))), (-((g / (lmbda * R)) + 1)))
        mub = m / (rho * S * b)
        db = b / V0
        Cl = 2 * W / (rho * V0 ** 2 * S)  # Lift coefficient [ ]
        u = [8 * sin(0.5 * pi * x) for x in range(t)]
        print(u)

    elif type is "dutch roll":
        t = 2793
        Temp0 = -9.75 + 273.15
        V0 = 161
        hp0 = 7888.02
        rho = rho0 * pow(((1 + (lmbda * hp0 / Temp0))), (-((g / (lmbda * R)) + 1)))
        mub = m / (rho * S * b)
        db = b / V0
        Cl = 2 * W / (rho * V0 ** 2 * S)  # Lift coefficient [ ]
        u = [8 * sin(0.5 * pi*x) for x in range(t)]
        u = np.array(u)

    else:
        t = 2853
        Temp0 = -9 + 273.15
        V0 = 179
        hp0 = 7587.14
        rho = rho0 * pow(((1 + (lmbda * hp0 / Temp0))), (-((g / (lmbda * R)) + 1)))
        mub = m / (rho * S * b)
        db = b / V0
        Cl = 2 * W / (rho * V0 ** 2 * S)  # Lift coefficient [ ]
        x = [range(t)]
        u = [0.45]+[1]*(t-1)

    AP = np.mat([[(Cydb - 2 * mub) * db, 0, 0, 0], [0, -0.5 * db, 0, 0],
                    [0, 0, -2 * mub * Kxx ** 2 * db, 2 * mub * Kxz ** 2 * db ** 2],
                    [Cndb, 0, 2 * mub * Kxz ** 2 * db, -2 * mub * Kzz ** 2 * db ** 2]])
    AQ = np.mat([[-Cyb, -Cl, -Cyp, -(Cyr - 4 * mub)], [0, 0, -1, 0], [-Clb, 0, -Clp, -Clr], [-Cnb, 0, -Cnp, Cnr]])
    AR = np.mat([[-Cyda, -Cydr], [0, 0], [-Clda, -Cldr], [-Cnda, -Cndr]])
    C = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2 / db, 0], [0, 0, 0, 2 / db]])
    D = np.mat([[0, 0], [0, 0], [0, 0], [0, 0]])
    A, B = GetAB(AP, AQ, AR)
    stateSpace = ss(A, B, C, D, x0)
    eig,vec = stateSpace.get_eig()
    eig = [val*db for val in eig]
    #stateSpace.plot_resp(u)
    print(eig)


ShortPeriod1 = symmetric("SP1")

ShortPeriod2 = symmetric("SP2")

Phugoid = symmetric("Phugoid")

Spiral = Asymmetric("spiral")

dutchR = Asymmetric("dutch roll")

damping = Asymmetric("damping")

def Analytical():
    pass