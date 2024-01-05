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
Trelax_1 = 0.2


##################
# Plot
##################

tauC = [0.5E-9, 1.2e-9, 2E-9, 6E-9]
color = ["navy","orange","turquoise","tomato"]

## Big plot

fig1 = plt.figure(figsize=(9,3.5))
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)

for i in range(len(tauC)):
    sim1 = cest_profile.calc_cr_eq(fqlist,Trelax_1,b1_1,b1_1*inhom,
                                          CS_A,CS_B,R2A,R2B,
                                          F_relax.rho_FH(tauC[i]),F_relax.sigma(tauC[i]),kAB,kBA)

    ax1.plot(fqlist[1:],sim1[1:]/sim1[0],c=color[i],linewidth=1,label="$\\tau_C$ = "+str(round(1E9*tauC[i],1))+" (ns)")
    ax2.plot(fqlist[1:],sim1[1:]/sim1[0],c=color[i],linewidth=1,label="$\\tau_C$ = "+str(round(1E9*tauC[i],1))+" (ns)")

ax1.plot([420,420,580,580,420],[0.55,0.9,0.9,0.55,0.55],ls="--",color="black",linewidth=0.4)
ax1.spines['top'].set_linewidth(0.5)
ax1.spines['right'].set_linewidth(0.5)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.get_xaxis().set_tick_params(pad=1)
ax1.get_yaxis().set_tick_params(pad=1)
ax1.tick_params(direction='out',axis='both',length=2,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=7)

ax1.locator_params(axis='x',nbins=8)
ax1.locator_params(axis='y',nbins=8)

ax1.set_ylabel('Normalized intensity',fontsize=8)
ax1.set_xlabel('Frequency [Hz]',fontsize=8)
ax1.legend(fontsize=8)

ax2.spines['top'].set_linewidth(0.5)
ax2.spines['right'].set_linewidth(0.5)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['bottom'].set_linewidth(0.5)
ax2.get_xaxis().set_tick_params(pad=1)
ax2.get_yaxis().set_tick_params(pad=1)
ax2.tick_params(direction='out',axis='both',length=2,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=7)

ax2.locator_params(axis='x',nbins=6)
ax2.locator_params(axis='y',nbins=6)
ax2.set_xlim(420,580)
ax2.set_ylim(0.55,0.9)
ax2.set_ylabel('Normalized intensity',fontsize=8)
ax2.set_xlabel('Frequency [Hz]',fontsize=8)
ax2.legend(fontsize=8)
plt.savefig("Plot_normalized.pdf")
