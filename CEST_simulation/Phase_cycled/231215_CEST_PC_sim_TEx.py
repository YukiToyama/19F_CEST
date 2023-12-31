# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import importlib
import cest_profile as cest_profile
import F_relax as F_relax
importlib.reload(cest_profile)
importlib.reload(F_relax)

##################
# Data
##################

b1_1 = 20
inhom = 0.1   
CS_A = 0
CS_B = 500  
kAB = 10  
kBA = 190 
R2A = 20 
R2B = 20
fqlist = np.arange(-300,910,1)
fqlist = np.append(12000,fqlist)

##################
# Plot
##################

Trelax = [0.05,0.1,0.2]
tauC = [0.5E-9, 1.2e-9, 2E-9, 6E-9]
color = ["navy","orange","turquoise","tomato"]
ls = [":","--","-"]
subplot = [221,222,223,224] 


fig1 = plt.figure(figsize=(7,6))

for k in range(len(tauC)):
    ax1 = fig1.add_subplot(subplot[k])
    
    for i in range(len(Trelax)):
        sim1 = cest_profile.calc_cr_pc(fqlist,Trelax[i],b1_1,b1_1*inhom,
                                              CS_A,CS_B,R2A,R2B,
                                              F_relax.rho_FH(tauC[k]),F_relax.sigma(tauC[k]),kAB,kBA)
        ax1.plot(fqlist[1:],sim1[1:]/sim1[0],c=color[k],ls=ls[i],linewidth=1,label="$T_{Ex}$ = "+str(int(Trelax[i]*1000))+" (ms)")

    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.get_xaxis().set_tick_params(pad=1)
    ax1.get_yaxis().set_tick_params(pad=1)
    ax1.tick_params(direction='out',axis='both',length=2,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=7)
    ax1.set_title("$\\tau_C$ = "+str(round(1E9*tauC[k])) +" [ns]",fontsize=10)
    ax1.locator_params(axis='x',nbins=6)
    ax1.locator_params(axis='y',nbins=6)
    
    ax1.set_ylim(0,1)
    ax1.set_ylabel('Normalized intensity',fontsize=7)
    ax1.set_xlabel('Frequency [Hz]',fontsize=7)
    ax1.yaxis.major.formatter._useMathText = True       
    ax1.legend(fontsize=6)

plt.tight_layout()
plt.savefig("Plot_PC.pdf")

