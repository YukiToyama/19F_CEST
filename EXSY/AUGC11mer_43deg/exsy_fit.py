# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:42:14 2019

@author: toyam
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

##################
# Function
##################

def exsy(t,kAB,kBA,RA,RB,M0):  
    pA = kBA/(kAB+kBA)
    pB = kAB/(kAB+kBA)
    a11 = RA + kAB
    a12 = -kBA
    a21 = -kAB
    a22 = RB + kBA
    rhamda1 = 0.5*((a11+a22)+((a11-a22)**2+4*kAB*kBA)**0.5)
    rhamda2 = 0.5*((a11+a22)-((a11-a22)**2+4*kAB*kBA)**0.5)
    IAA = M0*pA*(-1*(rhamda2-a11)*np.exp(-rhamda1*t)+(rhamda1-a11)*np.exp(-rhamda2*t))/(rhamda1-rhamda2)
    IBB = M0*pB*(-1*(rhamda2-a22)*np.exp(-rhamda1*t)+(rhamda1-a22)*np.exp(-rhamda2*t))/(rhamda1-rhamda2)
    IAB = M0*pA*(a21*np.exp(-rhamda1*t)-a21*np.exp(-rhamda2*t))/(rhamda1-rhamda2)
    IBA = M0*pB*(a12*np.exp(-rhamda1*t)-a12*np.exp(-rhamda2*t))/(rhamda1-rhamda2)
    return IAA, IBB, IAB, IBA  

##################
# Data
##################

data = pd.read_csv("data.csv")
tau = 0.003 

delay = np.loadtxt("vclist")*tau+920E-6
AA_exp = np.array(data[(data['assignment']=='dsRNA')].amp)
BB_exp = np.array(data[(data['assignment']=='ssRNA')].amp)
AB_exp = np.array(data[(data['assignment']=='dsRNAtossRNA')].amp)
BA_exp = np.array(data[(data['assignment']=='ssRNAtodsRNA')].amp)

AA_err = np.array(data[(data['assignment']=='dsRNA')].amp_err)
BB_err = np.array(data[(data['assignment']=='ssRNA')].amp_err)
AB_err = np.array(data[(data['assignment']=='dsRNAtossRNA')].amp_err)
BA_err = np.array(data[(data['assignment']=='ssRNAtodsRNA')].amp_err)

##################
# Fitting
##################

# Initial intensity
fit_params = Parameters()
fit_params.add('M0',value=7.5E11)

# rate constants
fit_params.add('kAB',value=80)
fit_params.add('kBA',value=40)

# Relaxation
fit_params.add('RA',value=5)
fit_params.add('RB',expr="RA")

def objective(fit_params):
    
    M0 = fit_params['M0'] 
    
    kAB = fit_params['kAB'] 
    kBA = fit_params['kBA'] 
    
    
    RA = fit_params['RA']
    RB = fit_params['RB']
    
    residual = np.zeros(0,dtype=float)
    
    for i in range(len(delay)):
      Adiag, Bdiag, ABex, BAex, = exsy(delay[i],kAB,kBA,RA,RB,M0)
      residual = np.append(residual, Adiag - AA_exp[i])
      residual = np.append(residual, Bdiag - BB_exp[i]) 
      residual = np.append(residual, ABex - AB_exp[i])
      residual = np.append(residual, BAex - BA_exp[i])
    return residual 


result = minimize(objective,fit_params,method="leastsq")

print(fit_report(result))
with open("report.txt", 'a') as fh:
    fh.write("\n\n#######################\n")
    fh.write("Best fit ")
    fh.write("\n#######################\n\n")
    fh.write(fit_report(result))
    
RMSD = np.sqrt(result.chisqr/result.ndata)

##################
# Best fit data
##################
# Calculate best fit curve
opt_params = result.params

opt_M0 = opt_params['M0'].value  

opt_kAB = opt_params['kAB'].value  
opt_kBA = opt_params['kBA'].value
opt_kex = opt_kAB+opt_kBA

opt_pA = opt_kBA/opt_kex
opt_pB = opt_kAB/opt_kex

opt_RA = opt_params['RA'].value 
opt_RB = opt_params['RB'].value 

AA_sim = np.zeros(len(delay))
BB_sim = np.zeros(len(delay))
AB_sim = np.zeros(len(delay))
BA_sim = np.zeros(len(delay))

for i in range(len(delay)):
  AA_sim[i], BB_sim[i], AB_sim[i], BA_sim[i] = exsy(delay[i],opt_kAB,opt_kBA,opt_RA,opt_RB,opt_M0)

##################
# Update Parameters
##################
fit_params = Parameters()
fit_params.add('M0',value=opt_M0)

# rate constants
fit_params.add('kAB',value=opt_kAB)
fit_params.add('kBA',value=opt_kBA)

# Relaxation
fit_params.add('RA',value=opt_RA)
fit_params.add('RB',expr="RA")


##################
# MC iteration
##################

col1 = ['kAB','kBA','kex','pA','pB','RA','RB','M0']
params_df = pd.DataFrame(columns=col1)
cycle = 1000

for k in range(cycle):
    noise = np.random.normal(0,RMSD,result.ndata)
    
    def MC(fit_params):
        
        M0 = fit_params['M0'] 
        
        kAB = fit_params['kAB'] 
        kBA = fit_params['kBA'] 
        
        
        RA = fit_params['RA']
        RB = fit_params['RB']
        
        residual = np.zeros(0,dtype=float)
        
        for i in range(len(delay)):
          Adiag, Bdiag, ABex, BAex, = exsy(delay[i],kAB,kBA,RA,RB,M0)
          residual = np.append(residual, Adiag - AA_sim[i])
          residual = np.append(residual, Bdiag - BB_sim[i]) 
          residual = np.append(residual, ABex - AB_sim[i])
          residual = np.append(residual, BAex - BA_sim[i])
        
        return residual + noise 
    
    result_temp = minimize(MC,fit_params,method="leastsq")  
    print(fit_report(result_temp))
    with open("report.txt", 'a') as fh:
        fh.write("\n\n#######################\n")
        fh.write("MC iterlation " + str(k))
        fh.write("\n#######################\n\n")
        fh.write(fit_report(result_temp))
    
    MC_params = result_temp.params
    
    MC_kAB = MC_params['kAB'].value  
    MC_kBA = MC_params['kBA'].value  
    MC_kex = MC_kAB+MC_kBA
    MC_pA = MC_kBA/MC_kex
    MC_pB = MC_kAB/MC_kex
    MC_RA = MC_params['RA'].value 
    MC_RB = MC_params['RB'].value 
    MC_M0 = MC_params['M0'].value 
    
    tmp_se = pd.Series([MC_kAB,MC_kBA,MC_kex,MC_pA,MC_pB,MC_RA,MC_RB,MC_M0],
                            index=params_df.columns)
    params_df = params_df.append(tmp_se, ignore_index=True)

params_df.to_csv("MC.csv")

####################################
# Output data frame
####################################
   
result_df = pd.DataFrame(columns=col1)

# Bestfit
tmp_se = pd.Series([opt_kAB,opt_kBA,opt_kex,opt_pA,opt_pB,opt_RA,opt_RB,opt_M0],
                        index=result_df.columns)
result_df = result_df.append(tmp_se, ignore_index=True)

# Error
# Read the MC parameters as a list
MC_kAB = np.array(params_df.kAB)
MC_kBA = np.array(params_df.kBA)
MC_kex = MC_kAB+MC_kBA
MC_pA = MC_kBA/MC_kex
MC_pB = MC_kAB/MC_kex
MC_RA = np.array(params_df.RA)
MC_RB = np.array(params_df.RB)
MC_M0 = np.array(params_df.M0)

tmp_se = pd.Series([np.std(MC_kAB),np.std(MC_kBA),np.std(MC_kex),np.std(MC_pA),np.std(MC_pB),
                    np.std(MC_RA),np.std(MC_RB),np.std(MC_M0)],
                        index=result_df.columns)
result_df = result_df.append(tmp_se, ignore_index=True)

result_df.to_csv("result.csv")

##################
# Calculate bestfit p/m 2*std curves 
##################

tplot = np.linspace(0,1.1*np.max(delay),1000)

AA_plot = np.zeros(len(tplot))
BB_plot = np.zeros(len(tplot))
AB_plot = np.zeros(len(tplot))
BA_plot = np.zeros(len(tplot))

for i in range(len(tplot)):
  AA_plot[i], BB_plot[i], AB_plot[i], BA_plot[i] = exsy(tplot[i],opt_kAB,opt_kBA,opt_RA,opt_RB,opt_M0)


MC_AA = np.zeros([len(MC_kAB),len(tplot)])
MC_BB = np.zeros([len(MC_kAB),len(tplot)])
MC_AB = np.zeros([len(MC_kAB),len(tplot)])
MC_BA = np.zeros([len(MC_kAB),len(tplot)])


for k in range(len(MC_kAB)):
  for i in range(len(tplot)):
    MC_AA[k,i], MC_BB[k,i], MC_AB[k,i], MC_BA[k,i] = exsy(tplot[i],MC_kAB[k],MC_kBA[k],MC_RA[k],MC_RB[k],MC_M0[k])
  
AA_top = AA_plot + np.std(MC_AA, axis=0)
AA_bottom = AA_plot - np.std(MC_AA, axis=0)
BB_top = BB_plot + np.std(MC_BB, axis=0)
BB_bottom = BB_plot - np.std(MC_BB, axis=0)

AB_top = AB_plot + np.std(MC_AB, axis=0)
AB_bottom = AB_plot - np.std(MC_AB, axis=0)
BA_top = BA_plot + np.std(MC_BA, axis=0)
BA_bottom = BA_plot - np.std(MC_BA, axis=0)


##################
# Plot
##################

cmap=['dodgerblue','turquoise','gray','purple']

Title = ['dsRNA', 'ssRNA', 'dsRNA to ssRNA', 'ssRNA to dsRNA']
Explist = [AA_exp, BB_exp, AB_exp, BA_exp]
Plotlist = [AA_plot, BB_plot, AB_plot, BA_plot]
Toplist = [AA_top, BB_top, AB_top, BA_top]
Bottomlist = [AA_bottom, BB_bottom, AB_bottom, BA_bottom]

Symbol = ['o','o','v','o']
ls = ['-','-','-','--']

fig1 = plt.figure(figsize=(4,3))
ax1 = fig1.add_subplot(111)

for i in range(len(Explist)):
  
  ax1.plot(tplot*1000,Plotlist[i], c=cmap[i],linewidth=1.,ls=ls[i])
  ax1.fill_between(tplot*1000,Toplist[i],Bottomlist[i],color=cmap[i],alpha=0.35)
  ax1.plot(delay*1000,Explist[i],c="white",markeredgecolor=cmap[i],marker=Symbol[i],markersize=3.5,ls="None",label=Title[i])
  
ax1.set_ylabel('Intensity',fontsize=8)
ax1.set_xlabel('Delay [ms]',fontsize=8)


ax1.spines['top'].set_linewidth(0.)
ax1.spines['right'].set_linewidth(0.)
ax1.spines['left'].set_linewidth(1)
ax1.spines['bottom'].set_linewidth(1)

ax1.get_xaxis().set_tick_params(length=1.5,pad=0.2,labelsize=8)
ax1.get_yaxis().set_tick_params(length=1.5,pad=0.2,labelsize=8)

ax1.tick_params(direction='out',axis='both',length=3,width=1,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=8)
ax1.locator_params(axis='x',nbins=8)
ax1.legend(fontsize=8)
plt.tight_layout()

plt.savefig("plot.pdf")
 

 
