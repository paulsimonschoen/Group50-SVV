import scipy as np

dc   = 1
db   = 1
muc  = 1
Czda = 1
Cmda = 1
Kyy  = 1
Cydb = 1
mub  = 1
Kxx  = 1
Kxz  = 1
Cxu  = 1
Cxa  = 1
Cz0  = 1
Cxq  = 1
Czu  = 1
Cza  = 1
Cx0  = 1
Cmu  = 1
cma  = 1
cmq  = 1
CYb  = 1
Cl   = 1
CYp  = 1
CYr  = 1
Clb  = 1
Clp  = 1
Clr  = 1
Cnb  = 1
Cnp  = 1
Cnr  = 1
Cxde = 1
Czde = 1
Cmde = 1
Cyda = 1
Cydr = 1
Clda = 1
Cldr = 1
Cnda = 1
Cndr = 1



SP  = np.matrix([[-2*muc*dc,0,0,0],[0,(Czda-2*muc)*dc,0,0],[0,0,-dc,0],[0,Cmda*dc,0,-2*muc*Kyy**2*dc**2]])
AP  = np.matrix([[(Cydb-2*mub)*db,0,0,0],[0,-0.5*db,0,0],[0,0,-2*mub*Kxx**2*db,2*mub*Kxz**2*db**2],[Cndb,0,2*mub*Kxz**2*db,-2*mub*Kzz**2*db**2]])
SQ  = np.matrix([[-Cxu,-Cxa,-Cz0,-Cxq],[-Czu,-Cza,Cx0,-(Czq+2muc)],[0,0,0,-1],[-Cmu,-Cma,0,-Cmq]])
AQ  = np.matrix([[-CYb,-Cl,-CYp,-(CYr-4*mub)],[0,0,-1,0],[-Clb,0,-Clp,-Clr],[-Cnb,0,-Cnp,Cnr]])
SR  = np.matrix([[-Cxde],[-Czde],[0],[Cmde]])
AR  = np.matrix([[-Cyda,-Cydr],[0,0],[-Clda,-Cldr],[-Cnda,-Cndr]])
SC  = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
AC  = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,2/db,0],[0,0,0,2/db]])
SD  = np.matrix([[0],[0],[0],[0]])
AD  = np.matrix([[0,0],[0,0],[0,0],[0,0]])

def GetAB(P,Q,R):
    A = P/Q
    B = P/R
    return A,B



