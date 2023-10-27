# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import importlib
import cest_profile as cest_profile
importlib.reload(cest_profile)

##################
# Data
##################

b1_1 = 20
inhom = 0.1   
CS_A = 0
CS_B = 500  
kAB = 10  
kBA = 190 
R1 = 3 
R2A = 20 
R2B = 20
fqlist = np.arange(-300,910,1)
fqlist = np.append(12000,fqlist)
Trelax_1 = 0.2

##################
# Plot
##################

sigma = [0,-3]
color = ["navy","tomato"]
ls = ["-","--"]

## Big plot

fig1 = plt.figure(figsize=(9,3.5))
ax1 = fig1.add_subplot(121)

for i in range(len(sigma)):
    sim1 = cest_profile.calc_cr_pc(fqlist,Trelax_1,b1_1,b1_1*inhom,
                                          CS_A,CS_B,R2A,R2B,
                                          R1,sigma[i],kAB,kBA)

    ax1.plot(fqlist[1:],sim1[1:]/sim1[0],c=color[i],ls=ls[i],linewidth=1,label="$\\sigma$ = "+str(sigma[i])+" (s$^{-1}$)")

ax1.plot([400,400,600,600,400],[0.3,0.55,0.55,0.3,0.3],ls="--",color="black",linewidth=0.4)
ax1.spines['top'].set_linewidth(0.5)
ax1.spines['right'].set_linewidth(0.5)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.get_xaxis().set_tick_params(pad=1)
ax1.get_yaxis().set_tick_params(pad=1)
ax1.tick_params(direction='out',axis='both',length=2,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=7)

ax1.locator_params(axis='x',nbins=6)
ax1.locator_params(axis='y',nbins=6)
ax1.set_ylim(0.0,0.65)
ax1.set_ylabel('Normalized intensity',fontsize=8)
ax1.set_xlabel('Frequency [Hz]',fontsize=8)     
ax1.legend(fontsize=8)

## close up
ax1 = fig1.add_subplot(122)

for i in range(len(sigma)):
    sim1 = cest_profile.calc_cr_pc(fqlist,Trelax_1,b1_1,b1_1*inhom,
                                          CS_A,CS_B,R2A,R2B,
                                          R1,sigma[i],kAB,kBA)

    ax1.plot(fqlist[1:],sim1[1:]/sim1[0],c=color[i],ls=ls[i],linewidth=1,label="$\\sigma$ = "+str(sigma[i])+" (s$^{-1}$)")

ax1.spines['top'].set_linewidth(0.5)
ax1.spines['right'].set_linewidth(0.5)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.get_xaxis().set_tick_params(pad=1)
ax1.get_yaxis().set_tick_params(pad=1)
ax1.tick_params(direction='out',axis='both',length=2,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=7)

ax1.locator_params(axis='x',nbins=6)
ax1.locator_params(axis='y',nbins=6)
ax1.set_xlim(400,600)
ax1.set_ylim(0.3,0.55)
ax1.set_ylabel('Normalized intensity',fontsize=8)
ax1.set_xlabel('Frequency [Hz]',fontsize=8)     
ax1.legend(fontsize=8)
plt.savefig("Plot_PC_normalized.pdf")