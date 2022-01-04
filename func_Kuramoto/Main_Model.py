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
    def phase_separation_mat(theta_t):
        a,b=np.meshgrid(theta_t, theta_t)
        complex_phase=(np.exp(1j*a)+np.exp(1j*b))
        r=abs(complex_phase)
        return 0.5*r
    @staticmethod
    def Compute_mean_freq(freq_table):
        N=int(freq_table.shape[1]/10)
        freq_dist=(freq_table[:, -N:].sum(axis=1))/N
        mean=np.mean(freq_dist)
        return mean, freq_dist
    @staticmethod
    def weight_dynamics(conn_mat, phase_diff, p_c, dt):
        p=phase_diff
        w=conn_mat
        p_c=np.full((phase_diff.shape[0], phase_diff.shape[1]), p_c)
        d_w=(p-p_c)*w*(1-w)
        np.fill_diagonal(d_w, 0)
        return dt*d_w
    @staticmethod
    def drop_neightbour(phase_mat, connections): #Drop node least in phase amongs neightbours, pick one most
        new_connections=np.copy(connections)     #in phase
    #Drop the neightbout you are least in fase
        for i in range(phase_mat.shape[0]):
            phases=phase_mat[i,:]
            edges=connections[i,:]
            conn_ind=np.where(edges==1)[0]
            conn_ind_0=np.where(edges==0)[0]
            most=np.argmax(phases[conn_ind_0])#Macimally in phase amongst non-neghtboours
            least=np.argmin(phases[conn_ind])#Maximally out of phase amongst neightbours
            most_ind=conn_ind_0[most]
            least_ind=conn_ind[least]
            new_connections[i,least_ind]=0 
            new_connections[least_ind,i]=0
            new_connections[i,most_ind]=1
            new_connections[most_ind,i]=1
        return new_connections
    @staticmethod
    def drop_randomly(connections):
        new_connections=np.copy(connections)
        for i in range(connections.shape[0]):
            edges=connections[i,:]
            conn_ind=np.where(edges==1)[0]
            conn_ind_0=np.where(edges==0)[0]
            choice_1=np.random.choice(conn_ind)
            choice_2=np.random.choice(conn_ind_0)
            new_connections[i,choice_1]=0 
            new_connections[choice_1,i]=0
            new_connections[i,choice_2]=1
            new_connections[choice_2,i]=1
        return new_connections      
    @staticmethod
    def drop_neightbour_forever(phase_mat, connections): #Drop the node least in phase reducing the number of edges
        new_connections=np.copy(connections)
        #Drop the neightbout you are least in fase
        for i in range(phase_mat.shape[0]):
            phases=phase_mat[i,:]
            edges=connections[i,:]
            conn_ind=np.where(edges==1)[0]
            least=np.argmin(phases[conn_ind])#Maximally out of phase amongst neightbours
            least_ind=conn_ind[least]
            new_connections[i,least_ind]=0 
            new_connections[least_ind,i]=0
        return new_connections
    
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
    def Hebbian_rewiring(self, timesteps, p_c, Uniform_start=True):
        #This function is for contionus change of weights
        self.p_c=p_c
        self.simulation_steps=timesteps
        self.freq_table=np.zeros([self.N, self.simulation_steps+1])
        self.theta_table=np.zeros([self.N, self.simulation_steps+1])
        self.theta_table[:,0]=self.init_theta
        self.freq_table[:,0]=self.init_freq
        self.theta=np.copy(self.init_theta)
        self.Hebbian_connectivity=np.zeros((self.N,self.N))+0.3
        for i in range(self.simulation_steps):
            d_theta=self.Update_non_uniform(self.init_freq,self.theta, self.dt, self.K, self.Hebbian_connectivity,
                        self.N)
            self.theta+=d_theta
            phase_diff=self.phase_separation_mat(self.theta)
            d_w=self.weight_dynamics(self.Hebbian_connectivity, phase_diff, self.p_c, self.dt)
            self.Hebbian_connectivity+=d_w
            for t in range(self.theta.shape[0]):
                if self.theta[t]>=self.lim:
                    self.theta[t]=self.theta[t]-self.lim
            self.theta_table[:,i+1]=self.theta
            self.freq_table[:,i+1]=d_theta/self.dt
        self.mean_freq, self.final_freq_dist=self.Compute_mean_freq(self.freq_table)
    def Hebbian_cycle(self, time_between, N_rewirings, *args, drop_forever=True, increase_coupling=True, fake=False, increment=0):
        #This function is for change of weights with regular rewirings
        self.time_between=time_between
        self.theta=np.copy(self.init_theta)
        self.conn_mat_list=[]
        self.increment=increment
        self.N_rewirings=N_rewirings
        self.drop_forever=drop_forever
        self.increase_coupling=increase_coupling
        self.total_steps=self.time_between*self.N_rewirings
        self.freq_table=np.zeros([self.N, self.total_steps+1])
        self.theta_table=np.zeros([self.N, self.total_steps+1])
        self.theta_table[:,0]=self.init_theta
        self.freq_table[:,0]=self.init_freq
        if len(args)>0:
            self.drop_seq=np.asarray(args[0])
            self.drop_seq=self.drop_seq*self.time_between
        #self.Update(10) #This update step is just rewuired to get theta started
        for i in range(self.total_steps):
            d_theta=self.Update_non_uniform(self.init_freq,self.theta, self.dt, self.K, self.connection_matrix,
                        self.N)
            self.theta+=d_theta
            for t in range(self.theta.shape[0]):
                if self.theta[t]>=self.lim:
                    self.theta[t]=self.theta[t]-self.lim
            self.theta_table[:,i+1]=self.theta
            self.freq_table[:,i+1]=d_theta/self.dt
            if i%1000==0 and i!=0:
                if fake==True:
                    con_mat=self.drop_randomly(self.connection_matrix)
                if fake==False:
                    phase_diff=self.phase_separation_mat(self.theta)
                    con_mat=self.drop_neightbour(phase_diff, self.connection_matrix)
                self.connection_matrix=con_mat
                self.conn_mat_list.append(con_mat)
                self.K+=self.increment
            if drop_forever==True:
                if i in self.drop_seq:
                    phase_diff=self.phase_separation_mat(self.theta)
                    con_mat=self.drop_neightbour_forever(phase_diff, self.connection_matrix)
                    self.connection_matrix=con_mat
                    self.conn_mat_list.append(con_mat)
        self.simulation_steps=self.total_steps+1 #We add one to include initial theta in order computation
        self.mean_freq, self.final_freq_dist=self.Compute_mean_freq(self.freq_table)
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