import control as ml
import matplotlib.pyplot as plt
import scipy as np


class State_Space:
    def __init__(self,A,B,C,D,x0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x0 = x0

    def get_ss(self):
        return ml.StateSpace(self.A,self.B,self.C,self.D)

    def get_eig(self):
        return np.linalg.eig(self.A)

    def get_response(self,u,t_):
        ss = self.get_ss()
        t,resp,xevo = np.signal.lsim((self.A,self.B,self.C,self.D),u,t_,self.x0)
        return t,resp

    def plot_resp(self,u,t_):
        fig = plt.figure()
        t,resp= self.get_response(u,t_)
        print(resp)
        plt.plot(t,resp)
        plt.show()










