# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import importlib
import matplotlib.pyplot as plt
import cest_matrix_3spin as cest_matrix_3spin
import F_relax as F_relax
importlib.reload(cest_matrix_3spin)
importlib.reload(F_relax)

##################
# Parameters
##################
outname = "3spin_standard_plot"
phasecycle = False
B1 = 20
inhom = 0.1   
CS_A = 0
CS_B = 500  
kAB = 10  
kBA = 190 
R2I = 20
Ieq = 0.94 # 19F
Seq = 1 # 1H

fqlist = np.arange(-300,910,1)

#################
# RF inhomogeneity 
#################

b1_frq = B1
b1_inh = B1*inhom
b1_inh_res = 10
b1_list = np.linspace(-2.0, 2.0, b1_inh_res) * b1_inh + b1_frq
b1_scales = sp.stats.norm.pdf(b1_list, b1_frq, b1_inh)

if phasecycle==True:
    b1_scales /= b1_scales.sum()
    b1_scales = b1_scales/2
else:
    b1_scales /= b1_scales.sum()
    
##################
# Plot
##################

Trelax = [0.05,0.1,0.2]
tauC = [0.5E-9, 1.2e-9, 2E-9, 6E-9]
color = ["navy","orange","turquoise","tomato"]

ls = [":","--","-"]
subplot = [221,222,223,224] 

fig1 = plt.figure(figsize=(7,6))

for i in range(len(tauC)):
    ax1 = fig1.add_subplot(subplot[i])

    for j in range(len(Trelax)):
        magI = np.zeros(len(fqlist))
        magS = np.zeros(len(fqlist))

        for k in range(len(b1_list)):
        
            w1Ix = b1_list[k]
            w1Iy = 0
        
            for l in range(len(fqlist)):
                
                carrier = fqlist[l]
                
                offsetA = CS_A - carrier
                offsetB = CS_B - carrier
                
                initial = cest_matrix_3spin.initial(kAB, kBA, Ieq, Seq)
                L = cest_matrix_3spin.L(offsetA, offsetB, w1Ix, w1Iy)
    
                R1I = F_relax.rho_FH(tauC[i])
                R1S = F_relax.rho_HF(tauC[i])+ F_relax.rho_HH(tauC[i])
                sigma_IS = F_relax.sigma(tauC[i])           
                
                Gamma = cest_matrix_3spin.Gamma(R1I, R1S, R2I, sigma_IS, kAB, kBA, Ieq, Seq)
                        
                rho = sp.linalg.expm((L+Gamma)*Trelax[j]) @ initial
                magI[l] += b1_scales[k]*rho[3]
                magS[l] += b1_scales[k]*rho[7]
                
                if phasecycle==True:
                    
                    initial2 = cest_matrix_3spin.initial2(kAB, kBA, Ieq, Seq)
                    rho = sp.linalg.expm((L+Gamma)*Trelax[j]) @ initial2
                    magI[l] -= b1_scales[k]*rho[3]
                    magS[l] -= b1_scales[k]*rho[7]
        
        ax1.plot(fqlist[1:],magI[1:]/initial[3],c=color[i],ls=ls[j],linewidth=1,label="$T_{Ex}$ = "+str(int(Trelax[j]*1000))+" (ms)")

    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.get_xaxis().set_tick_params(pad=1)
    ax1.get_yaxis().set_tick_params(pad=1)
    ax1.tick_params(direction='out',axis='both',length=2,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=7)
    ax1.set_title("$\\tau_C$ = "+str(round(1E9*tauC[i],2)) +" [ns]",fontsize=10)
    ax1.locator_params(axis='x',nbins=6)
    ax1.locator_params(axis='y',nbins=6)
    
    ax1.set_ylim(0,1.05)
    ax1.set_ylabel('Normalized intensity',fontsize=7)
    ax1.set_xlabel('Frequency [Hz]',fontsize=7)
    ax1.yaxis.major.formatter._useMathText = True       
    ax1.legend(fontsize=6)

plt.tight_layout()
plt.savefig(outname+".pdf")
