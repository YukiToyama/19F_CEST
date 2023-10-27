# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

# Offset and RF frequency are in Hz.

def L_2state(offsetA,offsetB,w1x,w1y,R2A,R2B,R1A,R1B,kAB,kBA):

    L = np.zeros([6,6])

    L[0,0] = -R2A-kAB
    L[0,1] = -offsetA*2*np.pi
    L[0,2] = w1y*2*np.pi
    L[0,3] = kBA
    
    L[1,0] = offsetA*2*np.pi
    L[1,1] = -R2A-kAB
    L[1,2] = -w1x*2*np.pi
    L[1,4] = kBA

    L[2,0] = -w1y*2*np.pi
    L[2,1] = w1x*2*np.pi
    L[2,2] = -R1A-kAB
    L[2,5] = kBA

    L[3,0] = kAB
    L[3,3] = -R2B-kBA
    L[3,4] = -offsetB*2*np.pi
    L[3,5] = w1y*2*np.pi
    
    L[4,1] = kAB
    L[4,3] = offsetB*2*np.pi
    L[4,4] = -R2B-kBA
    L[4,5] = -w1x*2*np.pi
    
    L[5,2] = kAB
    L[5,3] = -w1y*2*np.pi
    L[5,4] = w1x*2*np.pi
    L[5,5] = -R1B-kBA

    return L

def cpmg_calc(ncyc,T,p90,offsetA,offsetB,R2A,R2B,R1A,R1B,kAB,kBA):
    
    wcpmg = 0.25/p90 #in Hz
     
    # Rotation
    Lrot = L_2state(offsetA,offsetB,0,0,R2A,R2B,R1A,R1B,kAB,kBA)
    
    # x pulse
    Lx = L_2state(offsetA,offsetB,wcpmg,0,R2A,R2B,R1A,R1B,kAB,kBA)
    
    # y pulse
    Ly = L_2state(offsetA,offsetB,0,wcpmg,R2A,R2B,R1A,R1B,kAB,kBA)
    
    # Initial condition
    pA = kBA/(kAB+kBA)
    pB = kAB/(kAB+kBA)
    
    # Start from the Iy state
    I0 = np.zeros(6)
    I0[1] = pA
    I0[4] = pB
    
    signal = np.zeros(len(ncyc))
        
    # CPMG block
    for i in range(len(ncyc)):
        
        rho = I0
        
        COUNTER = 0
            
        while COUNTER < ncyc[i]:
        
            DELTA = T*0.25/ncyc[i] - p90
            rho =  sp.linalg.expm(DELTA*Lrot) @ rho
            rho =  sp.linalg.expm(2*p90*Ly) @ rho
            rho =  sp.linalg.expm(DELTA*Lrot) @ rho
            
            COUNTER += 1
            
            
        while COUNTER > 0:
            COUNTER -= 1
            rho =  sp.linalg.expm(DELTA*Lrot) @ rho
            rho =  sp.linalg.expm(2*p90*Ly) @ rho
            rho =  sp.linalg.expm(DELTA*Lrot) @ rho
            
        # Read y magnetization of the state A
        signal[i] = rho[1]
    
    return signal

