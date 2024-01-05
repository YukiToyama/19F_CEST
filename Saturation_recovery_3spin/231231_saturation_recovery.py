# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import importlib
import matplotlib.pyplot as plt
import F_relax as F_relax
importlib.reload(F_relax)

##################
# Parameters
##################
# Relaxation rates
tauC = 6.7e-9
R1I_CSA = 0.4
R1I = F_relax.rho_FH(tauC)+R1I_CSA 
R1S = F_relax.rho_HF(tauC)+ F_relax.rho_HH(tauC)
sigma = F_relax.sigma(tauC)   

# Equilibim 
Ieq = 0.94
Seq = 1

# Initial condition
initial = np.zeros(3,dtype=complex)
initial[0] = 0.5
initial[1] = 0
initial[2] = Seq

##################
# Read exoerimental data
##################
data,std = np.transpose(np.loadtxt("data_d1_10sec.txt"))
delay = np.loadtxt("vdlist")

Rapp = 2.55580075

##################
# Simulation
##################
Gamma=np.zeros([3,3])
Gamma[1,1] = -R1I
Gamma[2,2] = -R1S

# Cross relaxation
Gamma[1,2] = Gamma[2,1] = -sigma

# Return to thermal equilibrium
Gamma[1,0] = 2*(R1I*Ieq+sigma*Seq)
Gamma[2,0] = 2*(R1S*Seq+sigma*Ieq)
    
simt = np.linspace(0,1.1*np.max(delay),1000)
I_sim = np.zeros(len(simt))
S_sim = np.zeros(len(simt))

for i in range(len(simt)):
    rho = sp.linalg.expm(Gamma*simt[i])@initial
    I_sim[i] = rho[1]
    S_sim[i] = rho[2]

##################
# Plot
##################

fig1 = plt.figure(figsize=(4,3),dpi=300)
gs = fig1.add_gridspec(2,2,width_ratios=[1,1],height_ratios=[3,2])

ax1 = fig1.add_subplot(gs[0,0])

ax1.plot(simt,I_sim/Ieq, color='black',ls="-",linewidth=0.4,label="$I_z$ simulation")
ax1.plot(simt,1-np.exp(-Rapp*simt), color='tomato',ls="--",linewidth=0.7,label="Single exponential")

ax1.errorbar(delay,data,std, fmt='', color='black',
             ecolor='black', elinewidth=0.5, capsize=1.5, capthick=0.5, lw=0.)
ax1.plot(delay,data,markeredgewidth=0.5, color='white',linewidth=0.,
         markeredgecolor="black", marker='o', markersize=1.5,label="Experiment")

ax1.plot([0.2,0.2,3,3,0.2],[0.55,1.05,1.05,0.55,0.55],ls="--",color="black",linewidth=0.4)

ax1.set_xlim(0,5.2)
ax1.set_ylim(0,1.05)
ax1.spines['top'].set_linewidth(0.)
ax1.spines['right'].set_linewidth(0.)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.set_title("$I_z$ magnetization",fontsize=6)
ax1.set_ylabel('Normalized intensity',fontsize=6)  
ax1.set_xlabel('Delay [sec]',fontsize=6)     
ax1.get_xaxis().set_tick_params(pad=1)
ax1.get_yaxis().set_tick_params(pad=1)
ax1.yaxis.major.formatter._useMathText = True
ax1.tick_params(direction='out',axis='both',length=1,width=1,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax1.legend(fontsize=5)
plt.tick_params(labelsize=6,length=1)
plt.locator_params(axis='x',nbins=8)
plt.locator_params(axis='y',nbins=6)


ax2 = fig1.add_subplot(gs[0,1])
ax2.plot(simt,S_sim/Seq, color='black',ls="-",linewidth=0.4)

ax2.spines['top'].set_linewidth(0.)
ax2.spines['right'].set_linewidth(0.)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['bottom'].set_linewidth(0.5)
ax2.set_title("$S_z$ magnetization",fontsize=6)
ax2.set_ylabel('Normalized intensity',fontsize=6)  
ax2.set_xlabel('Delay [sec]',fontsize=6)   
ax2.set_xlim(0,5.2)  
ax2.set_ylim(0,1.05)
ax2.get_xaxis().set_tick_params(pad=1)
ax2.get_yaxis().set_tick_params(pad=1)
ax2.yaxis.major.formatter._useMathText = True
ax2.tick_params(direction='out',axis='both',length=1,width=1,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
plt.tick_params(labelsize=6,length=1)
plt.locator_params(axis='x',nbins=8)
plt.locator_params(axis='y',nbins=6)


ax3 = fig1.add_subplot(gs[1,0])

ax3.plot(simt,I_sim/Ieq, color='black',ls="-",linewidth=0.4,label="$I_z$ simulation")
ax3.plot(simt,1-np.exp(-Rapp*simt), color='tomato',ls="--",linewidth=0.5,label="Single exponential")

ax3.errorbar(delay,data,std, fmt='', color='black',
             ecolor='black', elinewidth=0.5, capsize=1.5, capthick=0.5, lw=0.)
ax3.plot(delay,data,markeredgewidth=0.5, color='white',linewidth=0.,
         markeredgecolor="black", marker='o', markersize=1.5,label="Experiment")


ax3.spines['top'].set_linewidth(0.5)
ax3.spines['right'].set_linewidth(0.5)
ax3.spines['left'].set_linewidth(0.5)
ax3.spines['bottom'].set_linewidth(0.5)
ax3.set_title("Dotted region",fontsize=6)
ax3.set_ylabel('Normalized intensity',fontsize=6)  
ax3.set_xlabel('Delay [sec]',fontsize=6)     
ax3.set_xlim(0.2,3)
ax3.set_ylim(0.55,1.05)
ax3.get_xaxis().set_tick_params(pad=1)
ax3.get_yaxis().set_tick_params(pad=1)
ax3.yaxis.major.formatter._useMathText = True
ax3.tick_params(direction='out',axis='both',length=1,width=1,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax3.legend(fontsize=5)
plt.tick_params(labelsize=6,length=1)
plt.locator_params(axis='x',nbins=8)
plt.locator_params(axis='y',nbins=6)

plt.tight_layout()
plt.savefig("saturation_recovery.pdf",transparent=True)