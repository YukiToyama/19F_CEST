# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

# Offset and RF frequency are in Hz.

def L(omegaA,omegaB,w1Ix,w1Iy,J):
    L=np.zeros([15,15])
    L[1,2] = -omegaA*2*np.pi
    L[1,3] = w1Iy*2*np.pi
    L[1,6] = -J*np.pi
    L[2,1] = omegaA*2*np.pi
    L[2,3] = -w1Ix*2*np.pi
    L[2,5] = J*np.pi
    L[3,1] = -w1Iy*2*np.pi
    L[3,2] = w1Ix*2*np.pi
    L[5,2] = -J*np.pi
    L[5,6] = -omegaA*2*np.pi
    L[5,7] = w1Iy*2*np.pi
    L[6,1] = J*np.pi
    L[6,5] = omegaA*2*np.pi
    L[6,7] = -w1Ix*2*np.pi
    L[7,5] = -w1Iy*2*np.pi
    L[7,6] = w1Ix*2*np.pi
    
    L[8,9] = -omegaB*2*np.pi
    L[8,10] = w1Iy*2*np.pi
    L[8,13] = -J*np.pi
    L[9,8] = omegaB*2*np.pi
    L[9,10] = -w1Ix*2*np.pi
    L[9,12] = J*np.pi
    L[10,8] = -w1Iy*2*np.pi
    L[10,9] = w1Ix*2*np.pi
    L[12,9] = -J*np.pi
    L[12,13] = -omegaB*2*np.pi
    L[12,14] = w1Iy*2*np.pi
    L[13,8] = J*np.pi
    L[13,12] = omegaB*2*np.pi
    L[13,14] = -w1Ix*2*np.pi
    L[14,12] = -w1Iy*2*np.pi
    L[14,13] = w1Ix*2*np.pi
    
    return L

def Gamma(R1I,R1S,R2I,sigma,kAB,kBA,Ieq,Seq):
    # KIN matrix is included.
    pA = kBA/(kAB+kBA)
    pB = 1-pA
    
    R2IS = R2I
    R1IS = R1I + R1S
    
    Gamma=np.zeros([15,15])
    Gamma[1,1] = -R2I-kAB
    Gamma[2,2] = -R2I-kAB
    Gamma[3,3] = -R1I-kAB
    Gamma[4,4] = -R1S-kAB
    Gamma[5,5] = -R2IS-kAB
    Gamma[6,6] = -R2IS-kAB
    Gamma[7,7] = -R1IS-kAB

    Gamma[8,8] = -R2I-kBA
    Gamma[9,9] = -R2I-kBA
    Gamma[10,10] = -R1I-kBA
    Gamma[11,11] = -R1S-kBA
    Gamma[12,12] = -R2IS-kBA
    Gamma[13,13] = -R2IS-kBA
    Gamma[14,14] = -R1IS-kBA
    
    Gamma[1,8] = kBA
    Gamma[2,9] = kBA
    Gamma[3,10] = kBA
    Gamma[4,11] = kBA
    Gamma[5,12] = kBA
    Gamma[6,13] = kBA
    Gamma[7,14] = kBA
    
    Gamma[8,1] = kAB
    Gamma[9,2] = kAB
    Gamma[10,3] = kAB
    Gamma[11,4] = kAB
    Gamma[12,5] = kAB
    Gamma[13,6] = kAB
    Gamma[14,7] = kAB


    # Cross relaxation (A state I spin - S spin)
    Gamma[3,4] = -sigma
    Gamma[4,3] = -sigma
    # Cross relaxation (B state I spin - S spin)
    Gamma[10,11] = -sigma
    Gamma[11,10] = -sigma
    
    # Return to thermal equilibrium
    # Note that population needs to be included.
    Gamma[3,0] = 2*pA*(R1I*Ieq+sigma*Seq)
    Gamma[4,0] = 2*pA*(R1S*Seq+sigma*Ieq)
    Gamma[10,0] = 2*pB*(R1I*Ieq+sigma*Seq)
    Gamma[11,0] = 2*pB*(R1S*Seq+sigma*Ieq)
    
    return Gamma

# Initial condition +z
def initial(kAB,kBA,Ieq,Seq):
    pA = kBA/(kAB+kBA)
    pB = 1-pA
    
    initial = np.zeros(15)
    initial[0] = 0.5
    initial[3] = pA*Ieq
    initial[4] = pA*Seq
    initial[10] = pB*Ieq
    initial[11] = pB*Seq
    
    return initial

# Initial condition -z
def initial2(kAB,kBA,Ieq,Seq):
    pA = kBA/(kAB+kBA)
    pB = 1-pA
    
    initial2 = np.zeros(15)
    initial2[0] = 0.5
    initial2[3] = -pA*Ieq
    initial2[4] = pA*Seq
    initial2[10] = -pB*Ieq
    initial2[11] = pB*Seq
    
    return initial2