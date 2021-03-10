# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:05:19 2020

@author: foadkrmn
"""

import sys
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
import bisect
import scipy.io as spio
from sklearn.neighbors.kde import KernelDensity

##########################################################################################
########### Select either RUL_AA_Initiation.mat or RUL_AA_Initiation_Mean.mat ############
file_name = '...\Codes\Matlab\RUL_AA_Initiation_Mean.mat'
mat = spio.loadmat(file_name, squeeze_me=True)
##########################################################################################

## Load measruement data
def load_d_mod(test):
    damage_index = mat['damage_index_modulus']
    loc = np.where(damage_index['test_name'][:] == test)
    Nf, q, m1, m2 = damage_index[loc][0][1], damage_index[loc][0][2], damage_index[loc][0][3], damage_index[loc][0][4]
    err = damage_index[loc][0][5]*1.5 ## *1.5 is to considering the systematic error as well 
    return Nf, q, m1, m2, err

def load_d_th(test):
    damage_index = mat['damage_index_th']
    loc = np.where(damage_index['test_name'][:] == test)
    N_th = damage_index[loc][0][1][:,0]
    D_th = damage_index[loc][0][1][:,1]
    err_th = damage_index[loc][0][2]*1.5   ## *1.5 is to considering the systematic error as well 
    return N_th, D_th, err_th

def load_d_inf(test):
    damage_index = mat['damage_index_inf']
    loc = np.where(damage_index['test_name'][:] == test)
    N_inf = damage_index[loc][0][1][:,0]
    D_inf =  damage_index[loc][0][1][:,1]
    err_inf = damage_index[loc][0][2]*1.2   ## *1.2 is to considering the systematic error as well 
    return N_inf, D_inf, err_inf

# State and parameter space functions
def state_space(D,m1,m2,q,Nf,delta_n,err,n):
    D += np.exp(np.random.normal(0,err,np.size(D)))*delta_n*(m1*(q/Nf)*(n/Nf)**(m1-1)+m2*((1-q)/Nf)*(n/Nf)**(m2-1))
    return D

def damage_state(Nf,q,m1,m2,cycle):
    D = q*(cycle/Nf)**m1 + (1-q)*(cycle/Nf)**m2
    return D
    
def parameter_space(p,h):
    mean = np.mean(p)
    var = np.var(p)
    mu, p_new = np.empty_like(p), np.empty_like(p)
    for i in range(len(p)):
        mu[i] = np.sqrt(1-h**2)*p[i] + (1 - np.sqrt(1-h**2))*mean
        p_new[i] = mu[i] + np.random.normal(0,h**2*var)
    return p_new

# Measurement effect on particles
def m_th(D,D_th,err_th):
    var = err_th # Mean squared error in thermodynamic ent damage index measurement from Modulus damage index
    PDF_m_th = ss.norm.pdf(D,D_th,var)
    return PDF_m_th

def m_inf(d,d_inf,err_inf):
    var = err_inf # Mean squared error in information ent damage index measurement from Modulus damage index
    PDF_m_inf = ss.norm.pdf(d,d_inf,var)
    return PDF_m_inf

# Resampling state and parameters
def resample_state(D,w): 
    w_norm  = np.sum(w) # Normalization factor for weights 
    w_ecdf = np.cumsum(w)/w_norm # New weight given the new measurement
    # Resample the points
    D_new, ind = np.empty_like(D), np.empty_like(D)
    for i,q in enumerate(D):
        ind[i] = bisect.bisect_left(w_ecdf, np.random.uniform(0, 1)) # Indexes for new samples 
        D_new[i] = D[int(ind[i])] # New weighted particles (samples) from previous step given new measuremnt
    # Regularize it!
#    std = np.std(D_new)
    bandwidth = 0.05 #1.06*std*len(D_new)**-0.2  ## used to be 0.08
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', algorithm='ball_tree') # Bandwidth = 0.006 is calculated based on Silverman's Rule of Thumb
    kde.fit(D_new[:,np.newaxis]) 
    return kde.sample(num_particles).flatten(), ind

def resample_parameters(p,ind,h): 
    p_re = np.empty_like(p)
    for i in range(len(ind)):
        p_re[i] = p[int(ind[i])]  
    return p_re    

def update_particles(D,Nf,m1,m2,q,h_nf,h_m1,h_m2,h_q,w):
    D, ind = resample_state(D,w)
    Nf = resample_parameters(Nf,ind,h_nf)
    m1 = resample_parameters(m1,ind,h_m1)
    m2 = resample_parameters(m2,ind,h_m2)
    q = resample_parameters(q,ind,h_q)
    return D, Nf, m1, m2, q

# Measure RUL given new observation and using particle weights
def measure_rul(D,m1,m2,q,Nf,delta_rul,i,D_threshold):
    D_rul, rul_est = np.empty_like(D), np.empty_like(D)
    for t in range(len(D)):
        if D[t] < D_threshold:
            i_rul = i
            D_rul[t] = D[t]
            while D_rul[t] < D_threshold and i_rul < np.max(Nf):
                i_rul += delta_rul
                D_rul[t] = damage_state(Nf[t],q[t],m1[t],m2[t],i_rul)
            rul_est[t] = i_rul
        elif D[t] > D_threshold:
            i_rul = i
            D_rul[t] = D[t]
            while D_rul[t] > D_threshold and i_rul > 0:
                i_rul -= delta_rul
                D_rul[t] = damage_state(Nf[t],q[t],m1[t],m2[t],i_rul)
            rul_est[t] = i_rul
        elif D[t] == D_threshold:
            rul_est[t] = i
    return rul_est

# Post processing functions
def average(p):
    p_avg = np.zeros(p.shape[1])
    for i in range(p.shape[1]):
        p_avg[i] = np.mean(p[:,i])
    return p_avg

def std(p):
    p_std = np.zeros(p.shape[1])
    for i in range(p.shape[1]):
        p_std[i] = np.std(p[:,i])
    return p_std

def median(p):
    p_med = np.zeros(p.shape[1])
    for i in range(p.shape[1]):
        p_med[i] = np.percentile(p[:,i],50)
    return p_med

def mean_rul_err(rul_est,true):
    rul_mean = np.mean(average(rul_est))
    print('Average of predictions: ',rul_mean)
    err = ((rul_mean - true)/true)*100
    print('Mean rul error : ', err,'%')
    
def rul_err(rul_est,true):
    rmse = np.sqrt(np.mean((average(rul_est)-true)**2))
    print('\nRUL est RMSE = ',rmse)

def true_damage(test,end,cycle):
    Nf, q, m1, m2, err = load_d_mod(test)
    D_true = np.empty(end)
    for i in range(end):
        D_true[i] = q*(cycle[i]/Nf)**m1+(1-q)*(cycle[i]/Nf)**m2   
    return D_true

def plot_parameters(Nf_hist,q_hist,m1_hist,m2_hist,test,update_cycle):
    lo_nf, lo_q, lo_m1, lo_m2 = np.empty(Nf_hist.shape[1]), np.empty(q_hist.shape[1]), np.empty(m1_hist.shape[1]), np.empty(m2_hist.shape[1])
    hi_nf, hi_q, hi_m1, hi_m2 = np.empty(Nf_hist.shape[1]), np.empty(q_hist.shape[1]), np.empty(m1_hist.shape[1]), np.empty(m2_hist.shape[1])
    for i in range(Nf_hist.shape[1]):
        lo_nf[i], lo_q[i] = np.percentile(Nf_hist[:,i],10), np.percentile(q_hist[:,i],10)
        lo_m1[i], lo_m2[i] = np.percentile(m1_hist[:,i],10), np.percentile(m2_hist[:,i],10)
        hi_nf[i], hi_q[i] = np.percentile(Nf_hist[:,i],90), np.percentile(q_hist[:,i],90)
        hi_m1[i], hi_m2[i] = np.percentile(m1_hist[:,i],90), np.percentile(m2_hist[:,i],90)
        
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(update_cycle.flatten(),average(Nf_hist),label='Mean', color='red', linewidth=2)
    ax[0,0].plot(update_cycle.flatten(),lo_nf, linestyle=':', color='red', label='90% CI')
    ax[0,0].plot(update_cycle.flatten(),hi_nf, linestyle=':', color='red')
    ax[0,0].set_ylim(bottom = np.min(Nf_hist[:,0]), top = np.max(Nf_hist[:,0]))
    ax[0,0].set_ylabel(r'$N_f$')
    ax[0,0].set_xlabel('Cycle')
    ax[0,0].axhline(y = load_d_mod(test)[0], linestyle='dashdot', label = 'True value', color='black')
    ax[0,0].legend()
    
    ax[1,0].plot(update_cycle.flatten(),average(q_hist),label='Mean', color='red', linewidth=2)    
    ax[1,0].plot(update_cycle.flatten(),lo_q, linestyle=':', color='red', label='90% CI')
    ax[1,0].plot(update_cycle.flatten(),hi_q, linestyle=':', color='red')
    ax[1,0].set_ylim(bottom = np.min(q_hist[:,0]), top = np.max(q_hist[:,0]))
    ax[1,0].set_ylabel('q')
    ax[1,0].set_xlabel('Cycle')
    ax[1,0].axhline(y = load_d_mod(test)[1], linestyle='dashdot', label = 'True value', color='black')
    ax[1,0].legend()
    
    ax[0,1].plot(update_cycle.flatten(),average(m1_hist),label='Mean', color='red', linewidth=2)
    ax[0,1].plot(update_cycle.flatten(),lo_m1, linestyle=':', color='red', label='90% CI')
    ax[0,1].plot(update_cycle.flatten(),hi_m1, linestyle=':', color='red')
    ax[0,1].set_ylim(bottom = np.min(m1_hist[:,0]), top = np.max(m1_hist[:,0]))
    ax[0,1].set_ylabel(r'$m_1$')
    ax[0,1].set_xlabel('Cycle')
    ax[0,1].axhline(y = load_d_mod(test)[2], linestyle='dashdot', label = 'True value', color='black')
    ax[0,1].legend()
    
    ax[1,1].plot(update_cycle.flatten(),average(m2_hist),label='Mean', color='red', linewidth=2)
    ax[1,1].plot(update_cycle.flatten(),lo_m2, linestyle=':', color='red', label='90% CI')
    ax[1,1].plot(update_cycle.flatten(),hi_m2, linestyle=':', color='red')
    ax[1,1].set_ylim(bottom = np.min(m2_hist[:,0]), top = np.max(m2_hist[:,0]))
    ax[1,1].set_ylabel(r'$m_2$')
    ax[1,1].set_xlabel('Cycle')
    ax[1,1].axhline(y = load_d_mod(test)[3], linestyle='dashdot', label = 'True value', color='black')
    ax[1,1].legend()
    
    fig.tight_layout()
    fig.show()

def plot_fill(D_hist,N_th,D_th,N_inf,D_inf,test,D_threshold):
    lo, hi, mid = np.empty(D_hist.shape[1]), np.empty(D_hist.shape[1]), np.empty(D_hist.shape[1])
    for i in range(D_hist.shape[1]):
        lo[i] = np.percentile(D_hist[:,i],10)
        hi[i] = np.percentile(D_hist[:,i],90)
        mid[i] = np.percentile(D_hist[:,i],50)
    cycle = np.linspace(1,D_hist.shape[1],D_hist.shape[1])
    D_true = true_damage(test,D_hist.shape[1],cycle)
    
    plt.figure()
    plt.fill_between(cycle,hi,lo, alpha=0.2)
#    plt.scatter(N_th,D_th, label = 'Thermo. Ent. Measurement')
    plt.scatter(N_th,D_th, marker="^", color='black', label = r'$D_s$')
#    plt.scatter(N_inf,D_inf, label = 'Inf. Ent. Measurement')
    plt.scatter(N_inf,D_inf, marker="o", color='green', label = r'$D_I$')
#    plt.plot(cycle,mid, color = 'red', label = 'Particle median')
    plt.plot(cycle,average(D_hist), color = 'red', label = 'Particle mean')
    plt.plot(cycle,D_true, color = 'black', linestyle='dashdot', label = 'True damage')
    plt.axhline(D_threshold, color='C7', linestyle='dotted', label ='Damage tolerance threshold')
    plt.ylim(bottom = 0, top = 1.6)
    plt.xlabel('Cycle')
    plt.ylabel('Damage index')
    plt.legend()
    plt.tight_layout()
#    plt.title(test)
    plt.show()

def plot_kde(obj,lo,hi,true,test):   
    obj_plot = np.linspace(lo,hi,10000)[:,np.newaxis]
    avg_std = np.mean(std(obj))
    bandwidth = 1.06*avg_std*len(obj)**-0.2
    plt.figure()
#    ax = plt.gca()
    for i in range(obj.shape[1]):
        a = obj[:,i][:,np.newaxis]
        #1.06*np.std(a)*len(a)**-0.2 # Bandwidth estimated by Silverman's Rule of Thumb
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', algorithm='ball_tree')     
        kde.fit(a)
        log_dens = kde.score_samples(obj_plot)     
        plt.plot(obj_plot,np.exp(log_dens))
#        vline_color = next(ax._get_lines.prop_cycler)['color']
#        plt.axvline(np.mean(a), linestyle=':', color = vline_color, label='Update %i' %(i+1))
    plt.axvline(np.mean(average(obj)), color = 'red', label='Mean of all predictions')
    plt.axvline(true, label='True value', linestyle='dashdot', color = 'black', linewidth=2)
    plt.ylabel('PDF')
    plt.xlabel('Cycle')
    plt.tight_layout()
    plt.legend()
#    plt.title('PF estimation - test %r' %test)
    
def rul_figure(N_th,N_inf,rul_est,N_init,test,D_threshold):
    N_all = np.concatenate((N_th,N_inf))
    N_all = np.sort(np.unique(N_all))
    Nf_est = average(rul_est)
    N_remain = Nf_est - N_all
    
    lo, hi = np.empty(rul_est.shape[1]), np.empty(rul_est.shape[1])
    for i in range(rul_est.shape[1]):
        lo[i] = np.percentile(rul_est[:,i],10) - N_all[i]
        hi[i] = np.percentile(rul_est[:,i],90) - N_all[i]
    
    x_axis, y_axis = np.linspace(0,N_init,N_init), np.linspace(N_init,0,N_init)

    plt.figure()
    plt.plot(x_axis, y_axis, color = 'black', linestyle='dashdot', label='True remaining life')
    plt.plot(N_all, N_remain, color = 'red', label = 'Estimated remaining life')
    plt.plot(N_all, hi, color = 'red', linestyle = ':', label = '90% CI')
    plt.plot(N_all, lo, color = 'red', linestyle = ':')
#    plt.plot([], [], ' ', label='Damage tolerance threshold = %.2f' %D_threshold)
    plt.xlabel('Cycle')
    plt.ylabel('RUL')
#    plt.grid()
    plt.legend()
#    plt.title('RUL estimation - test %r' %test)
    plt.tight_layout()
    plt.show
    

def excel(hist):
    obj = np.empty([hist.shape[1],3])
    for i in range(obj.shape[0]):
        obj[i,0] = np.percentile(hist[:,i],10)
        obj[i,1] = np.mean(hist[:,i])
        obj[i,2] = np.percentile(hist[:,i],90)
    return obj
    
################################         SELECT TEST            ##############################
# 5RAA01 (Overload), 5RAA05 (HCF), 5RAA07 (Programmed), 5RAA08 (Programmed), 5RAA11 (Overload), sample #
    
test = '5RAA07'

################################        Initialize variables    ###############################
num_particles = 1000 # Number of particles

D = np.abs(np.random.normal(0,0.01,num_particles))
err = load_d_mod(test)[-1] # Mean squared error in damage index measurement --> used as variance in state_space
delta_n = 1 # Step size in state space

D_threshold = 0.95 # Damage tolerance level at which RUL is measured. Max is 1, less gives more conservative measure of RUL

Nf = np.random.uniform(25000,32000,num_particles) # select initial values using load_d_mod(test)[0]
q = np.random.uniform(0.4,0.99,num_particles) # select initial values using load_d_mod(test)[1]
m1 = np.random.uniform(25,38,num_particles) # select initial values using load_d_mod(test)[2]
m2 = np.random.uniform(0.05,0.7,num_particles) # select initial values using load_d_mod(test)[3]

w = np.ones(len(D)) # initialize weights
delta_rul = 50 # Step size in RUL estimation
h_nf, h_q, h_m1, h_m2 =  0.005, 0.2, 0.1, 0.2 # h value in updating each parameter

################################        Load Measurement data    ###############################

N_th, D_th, err_th = load_d_th(test)
N_inf, D_inf, err_inf = load_d_inf(test)
end = int(load_d_mod(test)[0]*1.01)

################################          Prepare variables      ###############################
D_hist = np.zeros((num_particles,end))
rul_est = np.empty([num_particles,1])
Nf_hist, m1_hist, m2_hist, q_hist = Nf, m1, m2, q
update_cycle = np.array([0])

################################    START THE PARTICLE FILTERING    ###############################

for i in range(1,end):
    
    D = state_space(D,m1,m2,q,Nf,delta_n,err,i)
    
    if i in N_th and i in N_inf:
        w = m_th(D,D_th[np.where(N_th == i)[0][0]],err_th)*m_inf(D,D_inf[np.where(N_inf == i)[0][0]],err_inf)
        D, Nf, m1, m2, q = update_particles(D,Nf,m1,m2,q,h_nf,h_m1,h_m2,h_q,w)
        rul_est = np.column_stack([rul_est,measure_rul(D,m1,m2,q,Nf,delta_rul,i,D_threshold)])
        Nf_hist = np.column_stack([Nf_hist,Nf])
        m1_hist = np.column_stack([m1_hist,m1])
        m2_hist = np.column_stack([m2_hist,m2])
        q_hist = np.column_stack([q_hist,q])
        
        Nf, q = parameter_space(Nf,h_nf), parameter_space(q,h_q)
        m1, m2 = parameter_space(m1,h_m1), parameter_space(m2,h_m2)
        
        update_cycle = np.column_stack([update_cycle,i])
        
    elif i in N_th:
        w = m_th(D,D_th[np.where(N_th == i)[0][0]],err_th)
        D, Nf, m1, m2, q = update_particles(D,Nf,m1,m2,q,h_nf,h_m1,h_m2,h_q,w)
        rul_est = np.column_stack([rul_est,measure_rul(D,m1,m2,q,Nf,delta_rul,i,D_threshold)])
        Nf_hist = np.column_stack([Nf_hist,Nf])
        m1_hist = np.column_stack([m1_hist,m1])
        m2_hist = np.column_stack([m2_hist,m2])
        q_hist = np.column_stack([q_hist,q])
        
        Nf, q = parameter_space(Nf,h_nf), parameter_space(q,h_q)
        m1, m2 = parameter_space(m1,h_m1), parameter_space(m2,h_m2)
        
        update_cycle = np.column_stack([update_cycle,i])
        
    elif i in N_inf:
        w = m_inf(D,D_inf[np.where(N_inf == i)[0][0]],err_inf)
        D, Nf, m1, m2, q = update_particles(D,Nf,m1,m2,q,h_nf,h_m1,h_m2,h_q,w)
        rul_est = np.column_stack([rul_est,measure_rul(D,m1,m2,q,Nf,delta_rul,i,D_threshold)])
        Nf_hist = np.column_stack([Nf_hist,Nf])
        m1_hist = np.column_stack([m1_hist,m1])
        m2_hist = np.column_stack([m2_hist,m2])
        q_hist = np.column_stack([q_hist,q])
        
        Nf, q = parameter_space(Nf,h_nf), parameter_space(q,h_q)
        m1, m2 = parameter_space(m1,h_m1), parameter_space(m2,h_m2)
        
        update_cycle = np.column_stack([update_cycle,i])
    
    D_hist[:,i] = D
     
    sys.stdout.write("\r{0}".format((float(i)/end)*100))
    sys.stdout.flush()
        
# delete the arbitrary column in variable (first column)
rul_est = np.delete(rul_est, 0, 1)

### Show results:

rul_err(rul_est,load_d_mod(test)[0]) # print RMSE 
mean_rul_err(rul_est,load_d_mod(test)[0]) # print average rul

plot_parameters(Nf_hist,q_hist,m1_hist,m2_hist,test,update_cycle)

plot_fill(D_hist,N_th,D_th,N_inf,D_inf,test,D_threshold)

rul_figure(N_th,N_inf,rul_est,load_d_mod(test)[0],test,D_threshold)

bwidth = np.delete(bwidth, 0, 1)
update_cycle = np.delete(update_cycle, 0, 1)

rul_err(Nf_hist,load_d_mod(test)[0]) # print RMSE 
mean_rul_err(Nf_hist,load_d_mod(test)[0]) # print average rul

plot_kde(rul_est,24000,34000,load_d_mod(test)[0],test)
plot_kde(Nf_hist,15000,40000,load_d_mod(test)[0],test)
plot_kde(m1_hist,53,60,load_d_mod(test)[2],test)
plot_kde(m2_hist,0,1,load_d_mod(test)[3],test)
plot_kde(q_hist,0,1,load_d_mod(test)[1],test)
plot_kde(Nf[:,np.newaxis],20000,45000,load_d_mod(test)[0],test)


#### for excel file
excel_update_cycle = np.transpose(update_cycle)
excel_Nf_hist = excel(Nf_hist)
excel_q_hist = excel(q_hist)
excel_m1_hist = excel(m1_hist)
excel_m2_hist = excel(m2_hist)
excel_rul_est = excel(rul_est) - excel_update_cycle[1:len(excel_update_cycle)]
excel_D_hist = excel(D_hist)
cycle_excel = np.linspace(1,D_hist.shape[1],D_hist.shape[1])
excel_D_true = true_damage(test,D_hist.shape[1],cycle_excel)


