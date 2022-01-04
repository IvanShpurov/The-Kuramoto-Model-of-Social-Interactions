import numpy as np
import cmath
from numba import jit
class Kuramoto_model():
    lim=np.pi*2
    def __init__(self, coupling, initial_freq, initial_theta, dt, *args, Uniform_coupling=False, Hebbian_rewiring=False, 
                Kuramoto_Sagushi=False):
        self.K=coupling
        self.N=initial_theta.shape[0]
        self.init_freq=initial_freq
        self.init_theta=initial_theta
        if len(args)>0:
            self.connection_matrix=args[0]
        self.theta_table=None
        self.freq_table=None 
        self.Uniform=Uniform_coupling
        self.Hebbian=Hebbian_rewiring
        self.Sagushi=Kuramoto_Sagushi
        self.dt=dt
        self.mean_freq=0
        self.final_freq_dist=0
    #All static Methods are defined Here
    @staticmethod
    #Computes Critical coupling above whihc synchronisation would occur
    def Compute_crit_coupling(sigma):
        K_crit=np.sqrt(8/np.pi)*sigma
        return K_crit
    @staticmethod
    def Generate_initial_distribution(N, mean, dev):
        Initial_theta=np.around(np.random.uniform(0, 2*np.pi, N),2)
        Initial_freq=np.around(np.random.normal(mean, dev, N),2)
        return Initial_theta, Initial_freq
    @staticmethod
    @jit(nopython=True)
    def Update_uniform(Freq, theta, dt,K, N):
        d_theta=np.zeros_like(theta)
        for i in range(theta.shape[0]):
            cur_theta=theta[i]
            phi=np.array([np.sin(t-cur_theta) for t in theta])
            d_theta[i]=Freq[i]+(K/N)*phi.sum()
        return d_theta*dt
    @staticmethod
    @jit(nopython=True)
    def Update_non_uniform(Freq, theta, dt, K, C, N):
        N=theta.shape[0]
        d_theta=np.zeros_like(theta)
        for i in range(theta.shape[0]):
            cur_theta=theta[i]
            phi=np.array([np.sin(t-cur_theta) for t in theta])
            d_theta[i]=Freq[i]+(K/N)*phi@C[:,i]
        return d_theta*dt
    @staticmethod
    @jit(nopython=True)
    def Order_trig(theta_table):
        N=theta_table.shape[0]
        T=theta_table.shape[1]
        R=np.zeros(T)
        for i in range(T):
            theta=theta_table[:,i]
            cos=np.sum(np.cos(theta))
            sin=np.sum(np.sin(theta))
            r_2=(1/N**2)*(cos**2+sin**2)
            R[i]=r_2
        return np.sqrt(R)
    @staticmethod
    def Compute_mean_freq(freq_table):
        N=int(freq_table.shape[1]/10)
        freq_dist=(freq_table[:, -N:].sum(axis=1))/N
        mean=np.mean(freq_dist)
        return mean, freq_dist
    def Update(self, timesteps):
        self.simulation_steps=timesteps
        self.freq_table=np.zeros([self.N, self.simulation_steps+1])
        self.theta_table=np.zeros([self.N, self.simulation_steps+1])
        self.theta_table[:,0]=self.init_theta
        self.freq_table[:,0]=self.init_freq
        self.theta=np.copy(self.init_theta)
        if self.Uniform==True:
            for i in range(self.simulation_steps):
                d_theta=self.Update_uniform(self.init_freq,self.theta, self.dt, self.K, self.N)
                self.theta+=d_theta
                for t in range(self.theta.shape[0]):
                    if self.theta[t]>=self.lim:
                        self.theta[t]=self.theta[t]-self.lim
                self.theta_table[:,i+1]=self.theta
                self.freq_table[:,i+1]=d_theta/self.dt
        if self.Uniform==False:
            for i in range(self.simulation_steps):
                d_theta=self.Update_non_uniform(self.init_freq,self.theta, self.dt, self.K, self.connection_matrix,
                        self.N)
                self.theta+=d_theta
                for t in range(self.theta.shape[0]):
                    if self.theta[t]>=self.lim:
                        self.theta[t]=self.theta[t]-self.lim
                self.theta_table[:,i+1]=self.theta
                self.freq_table[:,i+1]=d_theta/self.dt
        self.simulation_steps+=1
        self.mean_freq, self.final_freq_dist=self.Compute_mean_freq(self.freq_table)
    def Hebbian_rewiring(self):
        pass
    def Hebbian_cycle(self):
        pass
    def Compute_order(self, Method="Trig"):
        self.R=np.zeros(self.simulation_steps)
        self.Phi=np.zeros(self.simulation_steps)
        if Method=="Complex":
            for i in range(self.simulation_steps):
                points_theta=list(self.theta_table[:,i])
                order=sum([np.exp(1j*p) for p in points_theta])/len(points_theta)
                r, phi=cmath.polar(order)
                self.R[i]=r
                self.Phi[i]=phi
        if Method=="Trig":
            self.R=self.Order_trig(self.theta_table)
        self.R_final=self.R[-1]