# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:48:59 2021

@author: toyam
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg


def Commute(A,B):
  return A@B-B@A

# This script allows to calculate the effect of spin lock up to 4 spins.
# No exchange, no relaxation
# Spin A, B, C, D
# A-> 2F, B-> H2, C -> H3, D -> H1

# Spin parameters
JAB = 52 #Hz
JAC = 21 #Hz
JAD = 17 #Hz
JBC = 11 #Hz
JBD = 0 #Hz

wAppm = 0. #ppm
wBppm = 5.2 #ppm
wCppm = 4.8 #ppm
wDppm = 6.3 #ppm

carrier = 5.0 #ppm
field = 600 # MHz

wRF = 12500 #Hz
p90 = 0.25/wRF

# Pauli matrices
Ix = 0.5*np.array([[0, 1],[1, 0]],dtype=complex)
Iy = 0.5*np.array([[0, -1j],[1j, 0]],dtype=complex)
Iz = 0.5*np.array([[1, 0],[0, -1]],dtype=complex)
E = np.eye(2)

# 4 successive kron product
def kron4(M1,M2,M3,M4):
  M3to4 = np.kron(M3,M4)
  M2to4 = np.kron(M2,M3to4)
  M1to4 = np.kron(M1,M2to4)
  return M1to4

# Operators
Ax = kron4(Ix,E,E,E)
Ay = kron4(Iy,E,E,E)
Az = kron4(Iz,E,E,E)

Bx = kron4(E,Ix,E,E)
By = kron4(E,Iy,E,E)
Bz = kron4(E,Iz,E,E)

Cx = kron4(E,E,Ix,E)
Cy = kron4(E,E,Iy,E)
Cz = kron4(E,E,Iz,E)

Dx = kron4(E,E,E,Ix)
Dy = kron4(E,E,E,Iy)
Dz = kron4(E,E,E,Iz)

# J Hamiltonian
HJAB = 2*np.pi*JAB*(Az@Bz) 
HJAC = 2*np.pi*JAC*(Az@Cz) 
HJAD = 2*np.pi*JAD*(Az@Dz) 
HJBC = 2*np.pi*JBC*(Bx@Cx+By@Cy+Bz@Cz) 
HJBD = 2*np.pi*JBD*(Bx@Dx+By@Dy+Bz@Dz) 

HJ = HJAB + HJAC + HJAD + HJBC + HJBD

# Chemical shift Hamiltonian
# Frequency (rad/s)
wA = wAppm*0.94*field*2*np.pi  # gF/gH ~ 0.94
wB = (wBppm-carrier)*field*2*np.pi
wC = (wCppm-carrier)*field*2*np.pi 
wD = (wDppm-carrier)*field*2*np.pi 

HCS = wA*Az + wB*Bz + wC*Cz  + wD*Dz 

# RF Hamiltonian on spin A
H_Ax = 2*np.pi*wRF*Ax

# Initial/observed state
initial =  Ax
obs = Ax 

## Function to calculate the spin evolution

def Lt(rho,H,t):
    
    Um = sp.linalg.expm(t*H*-1j)
    Up = sp.linalg.expm(t*H*1j)
    
    rho_evo = Um @ rho @ Up
    
    return rho_evo


#################
## CPMG
# (tau - 180x - tau )2n
#################

def cpmgsim(T):
    maxn = int(1000*T+0.1)
    nsync = np.arange(1,maxn,1)
    vcpmg = nsync/T
    tau = 0.25/vcpmg
    
    result_CPMG = np.zeros(len(nsync))
    
    for i in range(len(nsync)):
        rho = initial
    
        for k in range(int(nsync[i])):
            rho = Lt(rho,HCS+HJ,tau[i]-p90)
            rho = Lt(rho,HCS+HJ+H_Ax,2*p90)
            rho = Lt(rho,HCS+HJ,tau[i]-p90)
           
            rho = Lt(rho,HCS+HJ,tau[i]-p90)
            rho = Lt(rho,HCS+HJ+H_Ax,2*p90)
            rho = Lt(rho,HCS+HJ,tau[i]-p90)
            
        result_CPMG[i] = np.trace(rho @ obs)/np.trace(obs @ obs)
    
    I0 = np.trace(initial@obs)/np.trace(obs @ obs)
    Reff =  -1/T*np.log(result_CPMG/I0)
    return vcpmg,Reff


#############
# Plot
#############

fig1 = plt.figure(figsize=(2.5,1.5),dpi=300)
ax1 = fig1.add_subplot(111)

colorlist = ["black","dodgerblue","tomato"]
Tlist = [0.02,0.04,0.06]
for i in range(len(Tlist)):
    v,R = cpmgsim(Tlist[i])
    ax1.plot(v,R,linewidth=0.5, color=colorlist[i],label=str(int(1000*Tlist[i]))+" [ms]")
    
ax1.set_ylabel('$R_{2,eff}$ (s$^{-1}$)',fontsize=6)
ax1.set_xlabel('$\\nu_{CPMG}$ (Hz)',fontsize=6)
ax1.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax1.set_ylim(-0.02,0.1)

ax1.spines['top'].set_linewidth(0.)
ax1.spines['right'].set_linewidth(0.)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.legend(fontsize=6)

plt.tight_layout()
plt.savefig("cpmgsim.pdf")
