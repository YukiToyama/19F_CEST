# -*- coding: utf-8 -*-

import numpy as np

## Constants
hbar= 6.626E-34/2/np.pi
gH = 2.67522E8 # rad s-1 T-1
gF = 2.51815E8 # rad s-1 T-1

## Magnetic field
B0 = 14.1 # T

## Larmor frequencies
wH = B0*gH # rad s-1
wF = B0*gF # rad s-1

## 1H-19F distance
rHF1 = 2.5E-10 # m
rHF2 = 2.0E-10 # m
rHF4 = 2.7E-10 # m
rHF5 = 2.6E-10 # m

## 1H-19F effective distance
rHF = (rHF1**-6+rHF2**-6+rHF4**-6+rHF5**-6)**(-1/6)

## 1H-1H effective distance
rHH = 2.0E-10 # m

## Dipolar-dipolar interaction following Palmer's notation
## g1, g2: gyromagnetic ratio, r: effective distance
def Ad(r,g1,g2):
    return -1*np.sqrt(6)*1E-7*(hbar*g1*g2/r**3) 
    
## Spectral density function
def J(tauc,w):
    return 2/5*tauc/(1+w**2*tauc**2)

## 19F auto-relaxation due to 1H-19F dipolar interaction
def rho_FH(tauc):
    J_wFmwH =  J(tauc,wH-wF)
    J_wFpwH =  J(tauc,wH+wF)
    J_wF =  J(tauc,wF)
    return Ad(rHF,gH,gF)**2*(J_wFmwH+3*J_wF+6*J_wFpwH)/24 

## 1H auto-relaxation due to 1H-19F dipolar interaction
def rho_HF(tauc):
    J_wFmwH =  J(tauc,wH-wF)
    J_wFpwH =  J(tauc,wH+wF)
    J_wH =  J(tauc,wH)
    return Ad(rHF,gH,gF)**2*(J_wFmwH+3*J_wH+6*J_wFpwH)/24 

## 1H auto-relaxation due to 1H-1H(external) dipolar interaction
def rho_HH(tauc):
    J_0 =  J(tauc,0)
    J_wH =  J(tauc,wH)
    J_2wH =  J(tauc,2*wH)
    return Ad(rHH,gH,gH)**2*(J_0+3*J_wH+6*J_2wH)/24 

## 1H-19F cross-relaxation rate
def sigma(tauc):
    J_wFmwH =  J(tauc,wH-wF)
    J_wFpwH =  J(tauc,wH+wF)
    return Ad(rHF,gH,gF)**2*(-J_wFmwH+6*J_wFpwH)/24 
