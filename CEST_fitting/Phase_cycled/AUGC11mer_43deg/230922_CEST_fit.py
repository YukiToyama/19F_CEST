# -*- coding: utf-8 -*-

import numpy as np
from lmfit import minimize, Parameters, fit_report
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

major_offset = -732.36
minor_offset = 100

##################
# Parameters
##################

fit_params = Parameters()

# rate constants
fit_params.add('kAB',value=40,vary=True)
fit_params.add('kBA',value=100,vary=True)

# Relaxation
fit_params.add('R2A',value=20)
fit_params.add('R2B',value=20)
fit_params.add('R1',value=2)
fit_params.add('sigma',value=0,vary=False)

# CS
fit_params.add('CS_A',value=major_offset ,vary=False)
fit_params.add('CS_B',value=minor_offset)

##################
# Fitting
##################

def objective(fit_params):
    
    
    kAB = fit_params['kAB'] 
    kBA = fit_params['kBA']
    
    R2A = fit_params['R2A']
    R2B = fit_params['R2B']
    R1 = fit_params['R1']
    sigma = fit_params['sigma']
    CS_A = fit_params['CS_A']
    CS_B = fit_params['CS_B']
    
    residual = np.zeros(0)
    
    calc1 = cest_profile.calc_cr_pc(fq1,Trelax_1,b1_1,b1_1*inhom,CS_A,CS_B,R2A,R2B,R1,sigma,kAB,kBA)
    residual = np.append(residual, (data1[1:]/data1[0] - calc1[1:]/calc1[0]))
    
    calc2 = cest_profile.calc_cr_pc(fq2,Trelax_2,b1_2,b1_2*inhom,CS_A,CS_B,R2A,R2B,R1,sigma,kAB,kBA)
    residual = np.append(residual, (data2[1:]/data2[0] - calc2[1:]/calc2[0]))
    
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
opt_params = result.params

opt_CS_A = opt_params['CS_A'].value  
opt_CS_B = opt_params['CS_B'].value  
opt_deltaomega = np.abs(opt_CS_A-opt_CS_B)
opt_kAB = opt_params['kAB'].value  
opt_kBA = opt_params['kBA'].value  
opt_kex = opt_kAB+opt_kBA
opt_pA = opt_kBA/opt_kex
opt_pB = opt_kAB/opt_kex
opt_R1 = opt_params['R1'].value 
opt_sigma = opt_params['sigma'].value 
opt_R2A = opt_params['R2A'].value 
opt_R2B = opt_params['R2B'].value 

sim1 = cest_profile.calc_cr_pc(fq1,Trelax_1,b1_1,b1_1*inhom,
                                      opt_CS_A,opt_CS_B,opt_R2A,opt_R2B,
                                      opt_R1,opt_sigma,opt_kAB,opt_kBA)
sim2 = cest_profile.calc_cr_pc(fq2,Trelax_2,b1_2,b1_2*inhom,
                                      opt_CS_A,opt_CS_B,opt_R2A,opt_R2B,
                                      opt_R1,opt_sigma,opt_kAB,opt_kBA)
##################
# Update Parameters
##################

fit_params = Parameters()

# rate constants
fit_params.add('kAB',value=opt_kAB)
fit_params.add('kBA',value=opt_kBA)

# Relaxation
fit_params.add('R2A',value=opt_R2A)
fit_params.add('R2B',value=opt_R2B)
fit_params.add('R1',value=opt_R1)
fit_params.add('sigma',value=0,vary=False)

# CS
fit_params.add('CS_A',value=opt_CS_A,vary=False)
fit_params.add('CS_B',value=opt_CS_B)

####################################
# Output data frame
####################################
   
col1 = ['kAB','kBA','kex','pA','pB','R2A','R2B','R1','sigma','CS_A','CS_B','deltaomega']

result_df = pd.DataFrame(columns=col1)
tmp_se = pd.Series([opt_kAB,opt_kBA,opt_kex,opt_pA,opt_pB,
                    opt_R2A,opt_R2B,opt_R1,opt_sigma,opt_CS_A,opt_CS_B,opt_deltaomega],
                    index=result_df.columns)
result_df = result_df.append(tmp_se, ignore_index=True)


##################
# MC iteration
##################
params_df = pd.DataFrame(columns=col1)
cycle = 1000

for k in range(cycle):
    noise = np.random.normal(0,RMSD,result.ndata)
    
    def MC(fit_params):
        
        
        kAB = fit_params['kAB'] 
        kBA = fit_params['kBA']
        
        R2A = fit_params['R2A']
        R2B = fit_params['R2B']
        R1 = fit_params['R1']
        sigma = fit_params['sigma']
        CS_A = fit_params['CS_A']
        CS_B = fit_params['CS_B']
        
        residual = np.zeros(0)
        
        calc1 = cest_profile.calc_cr_pc(fq1,Trelax_1,b1_1,b1_1*inhom,CS_A,CS_B,R2A,R2B,R1,sigma,kAB,kBA)
        residual = np.append(residual, (sim1[1:]/sim1[0] - calc1[1:]/calc1[0]))
        
        calc2 = cest_profile.calc_cr_pc(fq2,Trelax_2,b1_2,b1_2*inhom,CS_A,CS_B,R2A,R2B,R1,sigma,kAB,kBA)
        residual = np.append(residual, (sim2[1:]/sim2[0] - calc2[1:]/calc2[0]))
        
        return residual  + noise 
    
    result_temp = minimize(MC,fit_params,method="leastsq")  
    print(fit_report(result_temp))
    with open("report.txt", 'a') as fh:
        fh.write("\n\n#######################\n")
        fh.write("MC iterlation " + str(k))
        fh.write("\n#######################\n\n")
        fh.write(fit_report(result_temp))
    
    opt_params = result_temp.params

    opt_CS_A = opt_params['CS_A'].value  
    opt_CS_B = opt_params['CS_B'].value  
    opt_deltaomega = np.abs(opt_CS_A-opt_CS_B)  
    opt_kAB = opt_params['kAB'].value  
    opt_kBA = opt_params['kBA'].value  
    opt_kex = opt_kAB+opt_kBA
    opt_pA = opt_kBA/opt_kex
    opt_pB = opt_kAB/opt_kex
    opt_R1 = opt_params['R1'].value 
    opt_sigma = opt_params['sigma'].value 
    opt_R2A = opt_params['R2A'].value 
    opt_R2B = opt_params['R2B'].value 
    
    tmp_se = pd.Series([opt_kAB,opt_kBA,opt_kex,opt_pA,opt_pB,
                        opt_R2A,opt_R2B,opt_R1,opt_sigma,opt_CS_A,opt_CS_B,opt_deltaomega],
                        index=params_df.columns)
    params_df = params_df.append(tmp_se, ignore_index=True)

params_df.to_csv("MC.csv")

####################################
# Output Error
####################################
   
# Read the MC parameters as a list
MC_kAB = np.array(params_df.kAB)
MC_kBA = np.array(params_df.kBA)
MC_kex = MC_kAB+MC_kBA
MC_pA = MC_kBA/MC_kex
MC_pB = MC_kAB/MC_kex
MC_R1 = np.array(params_df.R1)
MC_sigma = np.array(params_df.sigma)
MC_R2A = np.array(params_df.R2A)
MC_R2B = np.array(params_df.R2B)
MC_CS_A = np.array(params_df.CS_A)
MC_CS_B = np.array(params_df.CS_B)
MC_deltaomega = np.array(params_df.deltaomega)

tmp_se = pd.Series([np.std(MC_kAB),np.std(MC_kBA),np.std(MC_kex),np.std(MC_pA),np.std(MC_pB),
                    np.std(MC_R2A),np.std(MC_R2B),np.std(MC_R1),np.std(MC_sigma),
                    np.std(MC_CS_A),np.std(MC_CS_B),np.std(MC_deltaomega)],
                    index=result_df.columns)
result_df = result_df.append(tmp_se, ignore_index=True)
result_df.to_csv("result.csv")
