# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import cpmg as cpmg
importlib.reload(cpmg)

from lmfit import minimize, Parameters, fit_report

##############################
# Data import
##############################
p90 = 20E-6
Trelax = 0.02
ncyc = np.loadtxt("vclist")
data = (np.loadtxt("data.txt")).transpose()

##############################
# Fitting
##############################

vc = data[0]
intensity = data[1]
noise = data[2]

# Sort the data 
ncyc = vc[np.argsort(vc)]
exp = intensity[np.argsort(vc)]
error = noise[np.argsort(vc)]


# Spin parameters
fit_params = Parameters()

fit_params.add('offsetA',value=0,vary=False)
fit_params.add('R1A',value=2.00,vary=False)
fit_params.add('R2A',value=20,vary=True)

fit_params.add('offsetB',value=800,vary=True)
fit_params.add('R1B',expr="R1A")
fit_params.add('R2B',expr="R2A")

# Kinetics
fit_params.add('kAB',value=40,vary=True) 
fit_params.add('kBA',value=100,vary=True) 

# Scaling factor
fit_params.add('C',value=2E6,vary=True) 

def objective(fit_params):
    
    C = fit_params['C'] 
    
    offsetA = fit_params['offsetA'] 
    offsetB = fit_params['offsetB'] 

    R1A = fit_params['R1A'] 
    R1B = fit_params['R1B'] 
    
    R2A = fit_params['R2A'] 
    R2B = fit_params['R2B'] 
    
    kAB = fit_params['kAB']
    kBA = fit_params['kBA']
    
    calc = C*cpmg.cpmg_calc(ncyc,Trelax,p90,offsetA,offsetB,R2A,R2B,R1A,R1B,kAB,kBA)
    
    return exp - calc
    
result = minimize(objective,fit_params,method="leastsq")   
print(fit_report(result)) 
    

with open("report.txt", 'w') as fh:
    fh.write("\n\n#######################\n")
    fh.write("Best fit")
    fh.write("\n#######################\n\n")
    fh.write(fit_report(result))

RMSD = np.sqrt(result.chisqr/result.ndata)

##############################
# Best fit
##############################

opt_params = result.params

opt_offsetA = opt_params['offsetA'].value
opt_offsetB = opt_params['offsetB'].value
opt_deltaomega = np.abs(opt_offsetA-opt_offsetB)

opt_R1A = opt_params['R1A'].value
opt_R1B = opt_params['R1B'].value

opt_R2A = opt_params['R2A'].value
opt_R2B = opt_params['R2B'].value

opt_kAB = opt_params['kAB'].value
opt_kBA = opt_params['kBA'].value
opt_kex = opt_kAB+opt_kBA

opt_pA = opt_kBA/opt_kex
opt_pB = opt_kAB/opt_kex

opt_C = opt_params['C'].value  

sim = opt_C*cpmg.cpmg_calc(ncyc,Trelax,p90,opt_offsetA,opt_offsetB,
                           opt_R2A,opt_R2B,opt_R1A,opt_R1B,opt_kAB,opt_kBA)

nucpmg = (ncyc/Trelax)[1:]
R2eff_fit = -1/Trelax*np.log(sim[1:]/sim[0])
R2eff_exp = -1/Trelax*np.log(exp[1:]/exp[0])

##################
# Update Parameters
##################
fit_params = Parameters()

# Spin parameters
fit_params.add('offsetA',value=opt_offsetA,vary=False)
fit_params.add('R1A',value=opt_R1A,vary=False)
fit_params.add('R2A',value=opt_R2A,vary=True)
fit_params.add('offsetB',value=opt_offsetB,vary=True)
fit_params.add('R1B',expr="R1A")
fit_params.add('R2B',expr="R2A")

# Kinetics
fit_params.add('kAB',value=opt_kAB,vary=True) 
fit_params.add('kBA',value=opt_kBA,vary=True) 

# Scaling factor
fit_params.add('C',value=opt_C,vary=True) 



##################
# MC iteration
##################

col1 = ['kAB','kBA','kex','pA','pB','offsetA','offsetB','deltaomega',
        'R2A','R2B','R1A','R1B','M0']

params_df = pd.DataFrame(columns=col1)
cycle = 1000

for k in range(cycle):
    noise = np.random.normal(0,RMSD,result.ndata)
    
    def MC(fit_params):
        
        C = fit_params['C'] 
        
        offsetA = fit_params['offsetA'] 
        offsetB = fit_params['offsetB'] 
    
        R1A = fit_params['R1A'] 
        R1B = fit_params['R1B'] 
        
        R2A = fit_params['R2A'] 
        R2B = fit_params['R2B'] 
        
        kAB = fit_params['kAB']
        kBA = fit_params['kBA']
        
        calc = C*cpmg.cpmg_calc(ncyc,Trelax,p90,offsetA,offsetB,R2A,R2B,R1A,R1B,kAB,kBA)
        
        return sim - calc + noise
    
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
    MC_R2A = MC_params['R2A'].value 
    MC_R2B = MC_params['R2B'].value 
    MC_R1A = MC_params['R1A'].value 
    MC_R1B = MC_params['R1B'].value 
    MC_offsetA = MC_params['offsetA'].value 
    MC_offsetB = MC_params['offsetB'].value 
    MC_deltaomega = np.abs(MC_offsetA-MC_offsetB)
    MC_C = MC_params['C'].value 
    
    tmp_se = pd.Series([MC_kAB,MC_kBA,MC_kex,MC_pA,MC_pB,
                        MC_offsetA,MC_offsetB,MC_deltaomega,
                        MC_R2A,MC_R2B,MC_R1A,MC_R1B,MC_C],
                            index=params_df.columns)
    params_df = params_df.append(tmp_se, ignore_index=True)

params_df.to_csv("MC.csv")


####################################
# Output data frame
####################################
   
result_df = pd.DataFrame(columns=col1)
tmp_se = pd.Series([opt_kAB,opt_kBA,opt_kex,opt_pA,opt_pB,
                    opt_offsetA,opt_offsetB,opt_deltaomega,
                    opt_R2A,opt_R2B,opt_R1A,opt_R1B,opt_C],
                        index=result_df.columns)
result_df = result_df.append(tmp_se, ignore_index=True)

# Error
# Read the MC parameters as a list

MC_kAB = np.array(params_df.kAB)
MC_kBA = np.array(params_df.kBA)
MC_kex = MC_kAB+MC_kBA
MC_pA = MC_kBA/MC_kex
MC_pB = MC_kAB/MC_kex
MC_R2A = np.array(params_df.R2A)
MC_R2B = np.array(params_df.R2B)
MC_R1A = np.array(params_df.R1A)
MC_R1B = np.array(params_df.R1B)
MC_offsetA = np.array(params_df.offsetA)
MC_offsetB = np.array(params_df.offsetB)
MC_deltaomega = np.abs(MC_offsetA-MC_offsetB)
MC_C = np.array(params_df.M0)

tmp_se = pd.Series([np.std(MC_kAB),np.std(MC_kBA),np.std(MC_kex),np.std(MC_pA),np.std(MC_pB),
                    np.std(MC_offsetA),np.std(MC_offsetB),np.std(MC_deltaomega),
                    np.std(MC_R2A),np.std(MC_R2B),np.std(MC_R1A),np.std(MC_R1B),np.std(MC_C)],
                        index=result_df.columns)
result_df = result_df.append(tmp_se, ignore_index=True)

result_df.to_csv("result.csv")

##################
# Calculate bestfit p/m 2*std curves 
##################

MC_cpmg = np.zeros([len(MC_kAB),len(ncyc)])
MC_R2eff =np.zeros([len(MC_kAB),len(nucpmg)])

for k in range(len(MC_kAB)):
    MC_cpmg[k] = MC_C[k]*cpmg.cpmg_calc(ncyc,Trelax,p90,MC_offsetA[k],MC_offsetB[k],
                                          MC_R2A[k],MC_R2B[k],MC_R1A[k],MC_R1B[k],
                                          MC_kAB[k],MC_kBA[k]) 
    
    MC_R2eff[k] = -1/Trelax*np.log(MC_cpmg[k,1:]/MC_cpmg[k,0])

R2eff_top = R2eff_fit + np.std(MC_R2eff, axis=0)
R2eff_bottom = R2eff_fit - np.std(MC_R2eff, axis=0)


##################
# Plot
##################

fig1 = plt.figure(figsize=(3.4,2.6))
ax1 = fig1.add_subplot(111)
color = "black"

ax1.plot(nucpmg,R2eff_fit, c=color,linewidth=0.75,ls="-")
ax1.fill_between(nucpmg,R2eff_top,R2eff_bottom,color=color,alpha=0.35)
ax1.plot(nucpmg,R2eff_exp,c=color,markeredgecolor=color,marker="o",markersize=3.5,ls="None")

ax1.set_ylabel('$R_{2eff}$ [s$^{-1}$]',fontsize=8)
ax1.set_xlabel('vcpmg [Hz]',fontsize=8)


ax1.spines['top'].set_linewidth(0.)
ax1.spines['right'].set_linewidth(0.)
ax1.spines['left'].set_linewidth(1)
ax1.spines['bottom'].set_linewidth(1)

ax1.get_xaxis().set_tick_params(length=1.5,pad=0.2,labelsize=6)
ax1.get_yaxis().set_tick_params(length=1.5,pad=0.2,labelsize=6)

ax1.tick_params(direction='out',axis='both',length=3,width=1,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=8)
ax1.locator_params(axis='x',nbins=8)
plt.tight_layout()

plt.savefig("fitresult_CPMG.pdf")
 

    