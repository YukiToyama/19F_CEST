# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import importlib
import cest_profile as cest_profile
importlib.reload(cest_profile)

import pandas as pd


##################
# Data
##################

fq1,data1,error1 = np.loadtxt("10Hz.out").transpose()
fq2,data2,error2 = np.loadtxt("20Hz.out").transpose()

b1_1 = 11.32
b1_2 = 22.37
inhom = 0.1
Trelax_1 = 0.1
Trelax_2 = 0.1

o1p = -200 
sfo1 = 564.479
correction = -0.1089 # temperature correction

fq1ppm = fq1/sfo1+o1p+correction
fq2ppm = fq2/sfo1+o1p+correction


#########################
## Fit parameters
#########################
bestfit = pd.read_csv("result.csv")

kAB = bestfit.kAB[0]
kBA = bestfit.kBA[0]
R2A = bestfit.R2A[0]
R2B = bestfit.R2B[0]
R1 = bestfit.R1[0]
sigma = bestfit.sigma[0]
CS_A = bestfit.CS_A[0]
CS_B = bestfit.CS_B[0]
CS_Appm = CS_A/sfo1+o1p+correction
CS_Bppm = CS_B/sfo1+o1p+correction

MC = pd.read_csv("MC.csv")

MC_kAB = np.array(MC.kAB)
MC_kBA = np.array(MC.kBA)
MC_R2A = np.array(MC.R2A)
MC_R2B = np.array(MC.R2B)
MC_R1 = np.array(MC.R1)
MC_sigma = np.array(MC.sigma)
MC_CS_A = np.array(MC.CS_A)
MC_CS_B = np.array(MC.CS_B)


#########################
## Plot
#########################

bestfit_1 = cest_profile.calc_cr_eq_T0(fq1,Trelax_1,b1_1,b1_1*inhom,CS_A,CS_B,R2A,R2B,R1,sigma,kAB,kBA)
bestfit_2 = cest_profile.calc_cr_eq_T0(fq2,Trelax_2,b1_2,b1_2*inhom,CS_A,CS_B,R2A,R2B,R1,sigma,kAB,kBA)

MC_curve_1 = np.zeros([len(MC_kAB),len(fq1)])
MC_curve_2 = np.zeros([len(MC_kAB),len(fq2)])

for i in range(len(MC_kAB)):
    MC_curve_1[i] = cest_profile.calc_cr_eq_T0(fq1,Trelax_1,b1_1,b1_1*inhom,
                                            MC_CS_A[i],MC_CS_B[i],MC_R2A[i],
                                            MC_R2B[i],MC_R1[i],MC_sigma[i],MC_kAB[i],MC_kBA[i])
    MC_curve_2[i] = cest_profile.calc_cr_eq_T0(fq2,Trelax_2,b1_2,b1_2*inhom,
                                            MC_CS_A[i],MC_CS_B[i],MC_R2A[i],
                                            MC_R2B[i],MC_R1[i],MC_sigma[i],MC_kAB[i],MC_kBA[i])
    
top_1 = bestfit_1 + np.std(MC_curve_1, axis=0)
bottom_1 = bestfit_1 - np.std(MC_curve_1, axis=0)
top_2 = bestfit_2 + np.std(MC_curve_2, axis=0)
bottom_2 = bestfit_2 - np.std(MC_curve_2, axis=0)

color1="black" 
color2="tomato" 

fig1 = plt.figure(figsize=(4,3))
ax1 = fig1.add_subplot(111)

ax1.plot(fq1ppm[1:],bestfit_1[1:]/bestfit_1[0],linewidth=0.5,color=color1)

ax1.fill_between(fq1ppm[1:],top_1[1:]/bestfit_1[0],bottom_1[1:]/bestfit_1[0],color=color1,alpha=0.2)

ax1.plot(fq1ppm[1:],data1[1:]/data1[0],markeredgewidth=1, color=color1,linewidth=0.,
         markeredgecolor=color1, marker='o', markersize=1.)

ax1.plot(fq2ppm[1:],bestfit_2[1:]/bestfit_2[0],linewidth=0.5,color=color2)

ax1.fill_between(fq2ppm[1:],top_2[1:]/bestfit_2[0],bottom_2[1:]/bestfit_2[0],color=color2,alpha=0.2)

ax1.plot(fq2ppm[1:],data2[1:]/data2[0],markeredgewidth=1, color=color2,linewidth=0.,
         markeredgecolor=color2, marker='^', markersize=1.)

ymax = np.max(bestfit_1[1:]/bestfit_1[0])*1.1
ax1.plot([CS_Appm,CS_Appm],[0,ymax],ls="--",color='turquoise',linewidth=0.5)
ax1.plot([CS_Bppm,CS_Bppm],[0,ymax],ls="--",color='dodgerblue',linewidth=0.5)


ax1.set_ylabel("Normalized intensity",fontsize=8)
ax1.set_xlabel("Frequency",fontsize=8)
ax1.yaxis.major.formatter._useMathText = True

ax1.set_xlim(fq1ppm[1],fq1ppm[-1])

ax1.spines['top'].set_linewidth(0.)
ax1.spines['right'].set_linewidth(0.)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.get_xaxis().set_tick_params(pad=1)
ax1.get_yaxis().set_tick_params(pad=1)
ax1.tick_params(direction='out',axis='both',length=2,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=7)


plt.savefig("AUGC11mer_standard.pdf")