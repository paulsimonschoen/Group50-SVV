import control.matlab as ml
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
        return ml.ss(A,B,C,D)

    def get_eig(self):
        return np.eig(A)

    def get_response(self,u):
        ss = self.get_ss()
        t = np.arange(u)
        return ml.lsim(ss,u,t,self.x0),t

    def plot_resp(self,u):
        fig = plt.figure()
        resp,t = self.get_response(u)
        plt.plot(resp,t)
        plt.show()










