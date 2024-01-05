# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

# Offset and RF frequency are in Hz.

# Standard CEST without normalization.
def calc_cr_eq(fqlist,T,b1_frq,b1_inh,CS_A,CS_B,R2A,R2B,R1,sigma,kAB,kBA):
    
    # Initial condition
    pA = kBA/(kAB+kBA)
    pB = kAB/(kAB+kBA)
    
    I0 = np.zeros(7)
    I0[0] = 0.5
    I0[3] = pA
    I0[6] = pB
    
    mag = np.zeros(len(fqlist))
    
    L = np.zeros([7,7])
    # return to thermal equilibrium
    Ieq = 1 + sigma/R1/0.94  
    
    b1_inh_res = 5
    b1_list = np.linspace(-2.0, 2.0, b1_inh_res) * b1_inh + b1_frq
    b1_scales = sp.stats.norm.pdf(b1_list, b1_frq, b1_inh)
    b1_scales /= b1_scales.sum()
    
    for k in range(len(b1_list)):
        
        w1x = b1_list[k]
        w1y = 0
    
        for i in range(len(fqlist)):
            
            carrier = fqlist[i]
                    
            offsetA = CS_A - carrier
            offsetB = CS_B - carrier
            
            # R1 difference is not considered.
            R1A = R1
            R1B = R1
            
            L[1,1] = -R2A-kAB
            L[1,2] = -offsetA*2*np.pi
            L[1,3] = w1y*2*np.pi
            L[1,4] = kBA
            
            L[2,1] = offsetA*2*np.pi
            L[2,2] = -R2A-kAB
            L[2,3] = -w1x*2*np.pi
            L[2,5] = kBA
        
            L[3,0] = 2*R1A*pA*Ieq
            L[3,1] = -w1y*2*np.pi
            L[3,2] = w1x*2*np.pi
            L[3,3] = -R1A-kAB
            L[3,6] = kBA
        
            L[4,1] = kAB
            L[4,4] = -R2B-kBA
            L[4,5] = -offsetB*2*np.pi
            L[4,6] = w1y*2*np.pi
            
            L[5,2] = kAB
            L[5,4] = offsetB*2*np.pi
            L[5,5] = -R2B-kBA
            L[5,6] = -w1x*2*np.pi
            
            L[6,0] = 2*R1B*pB*Ieq
            L[6,3] = kAB
            L[6,4] = -w1y*2*np.pi
            L[6,5] = w1x*2*np.pi
            L[6,6] = -R1B-kBA
            
            rho = sp.linalg.expm(L*T) @ I0
            mag[i] += b1_scales[k]*rho[3]
    
    return mag

# Standard CEST normalized by the data recorded with TEx = 0.
def calc_cr_eq_T0(fqlist,T,b1_frq,b1_inh,CS_A,CS_B,R2A,R2B,R1,sigma,kAB,kBA):
    
    # Initial condition
    pA = kBA/(kAB+kBA)
    pB = kAB/(kAB+kBA)
    
    I0 = np.zeros(7)
    I0[0] = 0.5
    I0[3] = pA
    I0[6] = pB
    
    mag = np.zeros(len(fqlist))
    
    L = np.zeros([7,7])
    # return to thermal equilibrium
    Ieq = 1 + sigma/R1/0.94  
    
    b1_inh_res = 5
    b1_list = np.linspace(-2.0, 2.0, b1_inh_res) * b1_inh + b1_frq
    b1_scales = sp.stats.norm.pdf(b1_list, b1_frq, b1_inh)
    b1_scales /= b1_scales.sum()
    
    for k in range(len(b1_list)):
        
        w1x = b1_list[k]
        w1y = 0
    
        for i in range(len(fqlist)):
            
            carrier = fqlist[i]
                    
            offsetA = CS_A - carrier
            offsetB = CS_B - carrier
            
            # R1 difference is not considered.
            R1A = R1
            R1B = R1
            
            L[1,1] = -R2A-kAB
            L[1,2] = -offsetA*2*np.pi
            L[1,3] = w1y*2*np.pi
            L[1,4] = kBA
            
            L[2,1] = offsetA*2*np.pi
            L[2,2] = -R2A-kAB
            L[2,3] = -w1x*2*np.pi
            L[2,5] = kBA
        
            L[3,0] = 2*R1A*pA*Ieq
            L[3,1] = -w1y*2*np.pi
            L[3,2] = w1x*2*np.pi
            L[3,3] = -R1A-kAB
            L[3,6] = kBA
        
            L[4,1] = kAB
            L[4,4] = -R2B-kBA
            L[4,5] = -offsetB*2*np.pi
            L[4,6] = w1y*2*np.pi
            
            L[5,2] = kAB
            L[5,4] = offsetB*2*np.pi
            L[5,5] = -R2B-kBA
            L[5,6] = -w1x*2*np.pi
            
            L[6,0] = 2*R1B*pB*Ieq
            L[6,3] = kAB
            L[6,4] = -w1y*2*np.pi
            L[6,5] = w1x*2*np.pi
            L[6,6] = -R1B-kBA
            
            if abs(carrier)>10000:
                Trelax = 0
            else:
                Trelax = T
            
            rho = sp.linalg.expm(L*Trelax) @ I0
            mag[i] += b1_scales[k]*rho[3]
    
    return mag

# Phase-cycled CEST normalized by the data recorded with TEx = 0.
def calc_cr_pc(fqlist,T,b1_frq,b1_inh,CS_A,CS_B,R2A,R2B,R1,sigma,kAB,kBA):
    
    # Initial condition
    pA = kBA/(kAB+kBA)
    pB = kAB/(kAB+kBA)
    
    # Phase 1
    I0 = np.zeros(7)
    I0[0] = 0.5
    I0[3] = pA
    I0[6] = pB
    
    # Phase 2
    I0_2 = np.zeros(7)
    I0_2[0] = 0.5
    I0_2[3] = -pA
    I0_2[6] = -pB
    
    mag = np.zeros(len(fqlist))
    
    L = np.zeros([7,7])
    # return to thermal equilibrium
    Ieq = 1 + sigma/R1/0.94  
    
    b1_inh_res = 5
    b1_list = np.linspace(-2.0, 2.0, b1_inh_res) * b1_inh + b1_frq
    b1_scales = sp.stats.norm.pdf(b1_list, b1_frq, b1_inh)
    b1_scales /= b1_scales.sum()
    
    for k in range(len(b1_list)):
        
        w1x = b1_list[k]
        w1y = 0
    
        for i in range(len(fqlist)):
            
            carrier = fqlist[i]
                    
            offsetA = CS_A - carrier
            offsetB = CS_B - carrier
            
            # R1 difference is not considered.
            R1A = R1
            R1B = R1
            
            L[1,1] = -R2A-kAB
            L[1,2] = -offsetA*2*np.pi
            L[1,3] = w1y*2*np.pi
            L[1,4] = kBA
            
            L[2,1] = offsetA*2*np.pi
            L[2,2] = -R2A-kAB
            L[2,3] = -w1x*2*np.pi
            L[2,5] = kBA
        
            L[3,0] = 2*R1A*pA*Ieq
            L[3,1] = -w1y*2*np.pi
            L[3,2] = w1x*2*np.pi
            L[3,3] = -R1A-kAB
            L[3,6] = kBA
        
            L[4,1] = kAB
            L[4,4] = -R2B-kBA
            L[4,5] = -offsetB*2*np.pi
            L[4,6] = w1y*2*np.pi
            
            L[5,2] = kAB
            L[5,4] = offsetB*2*np.pi
            L[5,5] = -R2B-kBA
            L[5,6] = -w1x*2*np.pi
            
            L[6,0] = 2*R1B*pB*Ieq
            L[6,3] = kAB
            L[6,4] = -w1y*2*np.pi
            L[6,5] = w1x*2*np.pi
            L[6,6] = -R1B-kBA
            
            if abs(carrier)>10000:
                Trelax = 0
            else:
                Trelax = T
            
            rho = np.zeros(7)
            rho += sp.linalg.expm(L*Trelax) @ I0
            rho -= sp.linalg.expm(L*Trelax) @ I0_2
            mag[i] += b1_scales[k]*rho[3]
    
    return mag

