# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import importlib
import matplotlib.pyplot as plt
import cest_matrix_3spin as cest_matrix_3spin
import F_relax as F_relax
importlib.reload(cest_matrix_3spin)
importlib.reload(F_relax)

##############
# Paramaeters
##############
outname = "3spin_PC_normalized"
phasecycle = True

kex = 200
pB = 0.05
pA = 1-pB
kAB = kex*pB
kBA = kex-kAB

Ieq = 0.94 # 19F
Seq = 1 # 1H
tauC = [0.5E-9, 1.2e-9, 2E-9, 6E-9]
color = ["navy","orange","turquoise","tomato"]

B1 = 20
T = 0.2

fqlist = np.linspace(-300,910,400)
fqlist = np.append(-12000,fqlist)

CS_A = 0
CS_B = 500

R2I = 20

#################
# RF inhomogeneity 
#################

b1_frq = B1
b1_inh = B1*0.1
b1_inh_res = 10
b1_list = np.linspace(-2.0, 2.0, b1_inh_res) * b1_inh + b1_frq
b1_scales = sp.stats.norm.pdf(b1_list, b1_frq, b1_inh)

if phasecycle==True:
    b1_scales /= b1_scales.sum()
    b1_scales = b1_scales/2
else:
    b1_scales /= b1_scales.sum()

#################
# Simulation and Plot
#################
# Region to close up
lf = 430
hf = 560
lh = 0.55
hh = 0.80

fig1 = plt.figure(figsize=(8,6.5))
ax1 = fig1.add_subplot(221)
ax2 = fig1.add_subplot(222)
ax3 = fig1.add_subplot(224)

for j in range(len(tauC)):

    magI = np.zeros(len(fqlist))
    magS = np.zeros(len(fqlist))

    for k in range(len(b1_list)):
    
        w1Ix = b1_list[k]
        w1Iy = 0
    
        for i in range(len(fqlist)):
            
            carrier = fqlist[i]
            
            offsetA = CS_A - carrier
            offsetB = CS_B - carrier
            
            initial = cest_matrix_3spin.initial(kAB, kBA, Ieq, Seq)
            L = cest_matrix_3spin.L(offsetA, offsetB, w1Ix, w1Iy)

            R1I = F_relax.rho_FH(tauC[j])
            R1S = F_relax.rho_HF(tauC[j])+ F_relax.rho_HH(tauC[j])
            sigma_IS = F_relax.sigma(tauC[j])           
            
            Gamma = cest_matrix_3spin.Gamma(R1I, R1S, R2I, sigma_IS, kAB, kBA, Ieq, Seq)
                    
            rho = sp.linalg.expm((L+Gamma)*T) @ initial
            magI[i] += b1_scales[k]*rho[3]
            magS[i] += b1_scales[k]*rho[7]
            
            if phasecycle==True:
                initial2 = cest_matrix_3spin.initial2(kAB, kBA, Ieq, Seq)
                rho = sp.linalg.expm((L+Gamma)*T) @ initial2
                magI[i] -= b1_scales[k]*rho[3]
                magS[i] -= b1_scales[k]*rho[7]

      
    print(R1I, R1S, sigma_IS)    
    # Normalized to TEx = 0
    sigma = round(F_relax.sigma(tauC[j]),2)
    ax1.plot(fqlist[1:],magI[1:]/initial[3],label="$\\tau_C$ = "+str(round(tauC[j]*1e9,2))+" [ns]",color=color[j])
    # Normalized to off-resonance data
    ax2.plot(fqlist[1:],magI[1:]/magI[0],label="$\\tau_C$ = "+str(round(tauC[j]*1e9,2))+" [ns]",color=color[j])
    # Close up of the dotted region 
    ax3.plot(fqlist[1:],magI[1:]/magI[0],label="$\\tau_C$ = "+str(round(tauC[j]*1e9,2))+" [ns]",color=color[j])

ax1.spines['top'].set_linewidth(0.5)
ax1.spines['right'].set_linewidth(0.5)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.get_xaxis().set_tick_params(pad=1)
ax1.get_yaxis().set_tick_params(pad=1)
ax1.tick_params(direction='out',axis='both',length=2,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=7)
ax1.set_title("$T_{Ex}$="+str(round(1000*T))+" ms profile normalized to $T_{EX}$ = 0")
ax1.locator_params(axis='x',nbins=6)
ax1.locator_params(axis='y',nbins=6)
ax1.set_ylim(0.0,1.1)
ax1.set_ylabel('Intensity (normalized to TEx=0)',fontsize=8)  
ax1.set_xlabel('Frequency [Hz]',fontsize=8)     
ax1.legend(fontsize=8)

ax2.spines['top'].set_linewidth(0.5)
ax2.spines['right'].set_linewidth(0.5)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['bottom'].set_linewidth(0.5)
ax2.get_xaxis().set_tick_params(pad=1)
ax2.get_yaxis().set_tick_params(pad=1)
ax2.tick_params(direction='out',axis='both',length=2,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=7)
ax2.set_ylabel('Intensity (normalized to off-resonance)',fontsize=8)
ax2.set_title("Normalized to Off-resonance data")
ax2.plot([lf,lf,hf,hf,lf],[lh,hh,hh,lh,lh],ls="--",color="black",linewidth=1)
ax2.set_ylim(0.0,1.1)
ax2.locator_params(axis='x',nbins=6)
ax2.locator_params(axis='y',nbins=6)
ax2.set_xlabel('Frequency [Hz]',fontsize=8)     
ax2.legend(fontsize=8)

ax3.spines['top'].set_linewidth(0.5)
ax3.spines['right'].set_linewidth(0.5)
ax3.spines['left'].set_linewidth(0.5)
ax3.spines['bottom'].set_linewidth(0.5)
ax3.get_xaxis().set_tick_params(pad=1)
ax3.get_yaxis().set_tick_params(pad=1)
ax3.tick_params(direction='out',axis='both',length=2,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=7)
ax3.set_ylabel('Intensity (normalized to off-resonance)',fontsize=8)
ax3.set_title("Close-up of the dotted region")

ax3.locator_params(axis='x',nbins=6)
ax3.locator_params(axis='y',nbins=6)
ax3.set_xlim(lf,hf)
ax3.set_ylim(lh,hh)
ax3.set_xlabel('Frequency [Hz]',fontsize=8)     
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig(outname+".pdf")