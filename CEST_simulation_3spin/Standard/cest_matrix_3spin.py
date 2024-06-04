# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

# Offset and RF frequency are in Hz.

def L(omegaA,omegaB,w1Ix,w1Iy):
    L=np.zeros([8,8])
    L[1,2] = -omegaA*2*np.pi
    L[1,3] = w1Iy*2*np.pi
    L[2,1] = omegaA*2*np.pi
    L[2,3] = -w1Ix*2*np.pi
    L[3,1] = -w1Iy*2*np.pi
    L[3,2] = w1Ix*2*np.pi
    L[4,5] = -omegaB*2*np.pi
    L[4,6] = w1Iy*2*np.pi
    L[5,4] = omegaB*2*np.pi
    L[5,6] = -w1Ix*2*np.pi
    L[6,4] = -w1Iy*2*np.pi
    L[6,5] = w1Ix*2*np.pi
    return L

def Gamma(R1I,R1S,R2I,sigma,kAB,kBA,Ieq,Seq):
    # KIN matrix is included.
    pA = kBA/(kAB+kBA)
    pB = 1-pA
    
    Gamma=np.zeros([8,8])
    Gamma[1,1] = -R2I-kAB
    Gamma[2,2] = -R2I-kAB
    Gamma[3,3] = -R1I-kAB
    Gamma[4,4] = -R2I-kBA
    Gamma[5,5] = -R2I-kBA
    Gamma[6,6] = -R1I-kBA
    Gamma[7,7] = -R1S
    
    Gamma[1,4] = kBA
    Gamma[2,5] = kBA
    Gamma[3,6] = kBA
    Gamma[4,1] = kAB
    Gamma[5,2] = kAB
    Gamma[6,3] = kAB

    # Cross relaxation (A state I spin - S spin)
    Gamma[3,7] = -pA*sigma
    Gamma[7,3] = -sigma
    # Cross relaxation (B state I spin - S spin)
    Gamma[6,7] = -pB*sigma
    Gamma[7,6] = -sigma
    
    # Return to thermal equilibrium
    # Note that population needs to be included.
    Gamma[3,0] = 2*pA*(R1I*Ieq+sigma*Seq)
    Gamma[6,0] = 2*pB*(R1I*Ieq+sigma*Seq)
    Gamma[7,0] = 2*(R1S*Seq+sigma*Ieq)
    
    return Gamma

# Initial condition +z
def initial(kAB,kBA,Ieq,Seq):
    pA = kBA/(kAB+kBA)
    pB = 1-pA
    
    initial = np.zeros(8)
    initial[0] = 0.5
    initial[3] = pA*Ieq
    initial[6] = pB*Ieq
    initial[7] = Seq
    
    return initial

# Initial condition -z
def initial2(kAB,kBA,Ieq,Seq):
    pA = kBA/(kAB+kBA)
    pB = 1-pA
    
    initial2 = np.zeros(8)
    initial2[0] = 0.5
    initial2[3] = -pA*Ieq
    initial2[6] = -pB*Ieq
    initial2[7] = Seq
    
    return initial2