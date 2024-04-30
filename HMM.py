import matplotlib.pyplot as plt
import numpy as np

def gauss(x,mu,sigma):
    n = len(x)
    f = 1/np.sqrt((2*np.pi)^n * np.linalg.det(sigma)) * np.exp(-1/2 * (x - mu).T * np.linalg.inv(sigma) * (x - mu))
    return f

class Estep:
    def __init__(self,x,A,phi,pi):
        self.x = x
        self.A = A
        self.phi = phi
        self.pi = pi

    def forward_algorism(self):
        N = len(self.x)
        K = A.shape[0]

        a = np.zeros((N,K))
        a[0] = self.pi * self.phi[:,self.x[0]]

        for i in range(1,N):
            for j in range(K):
                a[i,j] = self.phi[j,x[i]] * np.sum(a[i-1] * self.A[:,j])

        return a

    def backward_algorism(self):
        N = len(self.x)
        K = A.shape[0]

        b = np.zeros((N,K))
        b[N-1] = 1

        for i in range(N-2,-1,-1): # -1のひとつ手前まで→0まで
            for j in range(K):
                b[i,j] = np.sum(b[i+1] * self.phi[:,x[i+1]] * self.A[j])
        
        return b

    def calc_Px(self):
        a = self.forward_algorism()
        Px = np.sum(a)
        return Px
    
class Mstep:
    def __init__(self,A,phi,pi):
        self.x = x
        self.A = A
        self.phi = phi
        self.pi = pi
    
    def Q(self,A_new,phi_new,pi_new):
        


A = np.array([[0.7,0.3],[0.4,0.6]])
phi = np.array([[0.1,0.4,0.5],[0.7,0.2,0.1]])
pi = np.array([0.6,0.4])

x = [0,1,2,0,2]

estep = Estep(x,A,phi,pi)

alpha = estep.forward_algorism()
print(alpha)
beta = estep.backward_algorism()
print(beta)
