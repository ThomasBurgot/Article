#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:15:46 2023

Stability and accuracy analysis program associated with the article: 
"Stability of constant and variable-coefficient semi-implicit schemes for the
fully elastic system of Euler equations in the case of steep slopes"
Thomas Burgot, Ludovic Auger, Pierre Bénard
MWR

Météo-France

Figures (1)--(9)


@author: tburgot
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import rc
from matplotlib import cm


#%% Constants
R = 287.0
Cp = 1004.0
Cv = Cp-R
g = 9.81

#%% Plot parameters

# Default parameters, increase them for a more accurate plot:
N_alpha = 38
NG = 31
    
    
#%% Default parameters
T_star = 350.0 # temperature of the implicit part
T_e = 350.0 # cold temperature of the implicit part in the vertical momentum equation

dx = 100.0 # horizontal resolution
Lx = 8000.0 # horizontal length of the domain
Nx = int(Lx/dx+1)-1 # Number of points on the horizontal direction
nk = np.linspace(-(Nx-1)/2,(Nx-1)/2,4) 
k = 2*np.pi/Lx*nk[-1:]
dz = 2.0 # height of the first level
G = np.linspace(0.0,3.0,NG) # slope
dt = np.array([3.0]) # time step

Limp=False; # implicit treatment of orographic term if True
Niter=2; # Niter of the ICI scheme
LSI2TL=False; # SI-2TL-E scheme 

Lfig='7' #choose the number of the figure of the article 
#('1','2','3','4','5a','5b','6','7','8','9')

if Lfig=='1':
    T_e = 350.0; Limp = False; dt = np.array([3.0]); LSI2TL=False; Niter=2
elif Lfig=='2':
    T_e = 100.0; Limp = False; dt = np.array([3.0]); LSI2TL=False; Niter=2
elif Lfig=='3':
    T_e = 2.0*10**(-5); Limp = False; dt = np.array([3.0]); LSI2TL=False; Niter=2
elif Lfig=='4':
    T_e = 100.0; Limp = False; dt = np.array([3.0]); LSI2TL=True; Niter=2
elif Lfig=='5a':
    T_e = 100.0; Limp = False; dt = np.array([3.0]); LSI2TL=False; Niter=4
elif Lfig=='5b':    
    T_e = 100.0; Limp = False; dt = np.array([3.0]); LSI2TL=False; Niter=8
elif Lfig=='6':
    T_e = 100.0; Limp = False; dt = np.array([3.0/10.5]); LSI2TL=False; Niter=2
elif Lfig=='7':
    T_e = 100.0; Limp = True; dt = np.array([3.0]); LSI2TL=False; Niter=2
elif Lfig=='8':
    T_e = 100.0; Limp = True; dt = np.array([3.0]); LSI2TL=False; Niter=4
elif Lfig=='9':
    T_e = 100.0; Limp = True; dt = np.array([3.0]); LSI2TL=True; Niter=2


r = T_e/T_star # ratio
alpha = np.linspace(-0.95,0.0,N_alpha) #thermal residual    
T_bar = alpha*T_star+T_star # temperature of the tangent-linear part
NT = len(T_bar)



#%% Physical quantities
H = R*T_star/g # characteristic height
N = g/np.sqrt(Cp*T_star) # Brunt Vaisala frequency


#%% Operators routines
def D_X(k,G,T_bar,i,ig,it):
    res = 1.0j*k[i]
    return (res)

def D_Xi(k,G,T_bar,i,ig,it):
    res = 1.0j*k[i]
    return (res)

def D_Xp(k,G,m,T_bar,i,ig,l,it):
    H_bar = R*T_bar[it]/g
    res = 1.0j*(k[i]+G[ig]*m[l]/H_bar)-G[ig]/2.0/H_bar
    return (res)

def D_Xpi(k,G,m,T_bar,i,ig,l,it):
    res = 1.0j*(k[i]+G[ig]*m[l]/H)-G[ig]/2.0/H
    return(res)

def operators(i,l,j,it,ig,Limp,k,m,G,dt):
    drond = 1.0j*m-0.5
    H_bar = R*T_bar[it]/g
    # Linear operator of the implicit problem (\mathcal{L^*} in the article):
    L = 1.0j*np.zeros((4,4))
    # Tangent-linear operator of the full problem (\bar{\mathcal{L}} in the article):
    M = 1.0j*np.zeros((4,4))            
    M[0,2] = R*D_Xp(k,G,m,T_bar,i,ig,l,it)
    M[0,3] = -R*T_bar[it]*(drond[l]+1.0)*D_Xp(k,G,m,T_bar,i,ig,l,it)
    M[1,3] = -g/H_bar*drond[l]*(drond[l]+1.0)-g*G[ig]*D_Xp(k,G,m,T_bar,i,ig,l,it)*(drond[l]+1.0)
    M[1,2] = g*G[ig]*D_Xp(k,G,m,T_bar,i,ig,l,it)/T_bar[it]
    M[2,0] = -R*T_bar[it]/Cv*D_X(k,G,T_bar,i,ig,it)
    M[2,1] = -R*T_bar[it]/Cv  
    M[3,0] = -Cp/Cv*D_X(k,G,T_bar,i,ig,it)*(drond[l]+1.0)+D_Xp(k,G,m,T_bar,i,ig,l,it)
    M[3,1] = -Cp/Cv*(drond[l]+1.0)
    if Limp:
        L[0,2] = R*D_Xpi(k,G,m,T_bar,i,ig,l,it)
        L[0,3] = -R*T_star*(drond[l]+1.0)*D_Xpi(k,G,m,T_bar,i,ig,l,it)
        L[1,3] = -g/r/H*drond[l]*(drond[l]+1.0)-g*G[ig]*D_Xpi(k,G,m,T_bar,i,ig,l,it)*(drond[l]+1.0)
        L[1,2] = g*G[ig]*D_Xpi(k,G,m,T_bar,i,ig,l,it)/T_star
        L[2,0] = -R*T_star/Cv*D_Xi(k,G,T_bar,i,ig,it)
        L[2,1] = -R*T_star/Cv  
        L[3,0] = -Cp/Cv*D_Xi(k,G,T_bar,i,ig,it)*(drond[l]+1.0)+D_Xpi(k,G,m,T_bar,i,ig,l,it)
        L[3,1] = -Cp/Cv*(drond[l]+1.0)    
    else:
        L[0,2] = R*D_X(k,G,T_bar,i,ig,it)
        L[0,3] = -R*T_star*(drond[l]+1.0)*D_X(k,G,T_bar,i,ig,it)
        L[1,3] = -g/r/H*drond[l]*(drond[l]+1.0)
        L[2,0] = -R*T_star/Cv*D_X(k,G,T_bar,i,ig,it)
        L[2,1] = -R*T_star/Cv  
        L[3,0] = -Cp/Cv*D_X(k,G,T_bar,i,ig,it)*(drond[l]+1.0)+D_X(k,G,T_bar,i,ig,it)
        L[3,1] = -Cp/Cv*(drond[l]+1.0)     
    L[0,:] = L[0,:]/drond[l]
    L[3,:] = L[3,:]/(drond[l]+1.0)
    M[0,:] = M[0,:]/drond[l]
    M[3,:] = M[3,:]/(drond[l]+1.0)
    return(M,L)
    
    
def ICI(Limp,LSI2TL,k,G,dt,Niter):
    Nk=len(k); Nt=len(dt);
    m = 2.0*np.pi*np.array([np.exp(0.2*j) for j in range(0,40)]) #\nu in the article
    Nm = len(m)
    lmax = np.zeros((Nt,Nk,Nm,NT,NG)) #growth rate 
    rmax = np.zeros((Nt,Nk,Nm,NT,NG)) #epsilon
    for i in range(Nk):
      for it in range(NT):
        for j in range(Nt):
          for l in range(Nm):
            for ig in range(NG):
               (M,L)=operators(i,l,j,it,ig,Limp,k,m,G,dt)
               SI = np.eye(4)-dt[j]/2.0*L
               EX = np.eye(4)+dt[j]/2.0*L
               Re = dt[j]/2.0*(M-L)
               invSI = np.linalg.inv(SI)
               A = np.matmul(invSI,EX+Re)
               B = np.matmul(invSI,Re)
               if LSI2TL:
                   AA = 1.0j*np.zeros((8,8))    
                   BB = 1.0j*np.zeros((8,8))
                   AA[0:4,0:4] = np.eye(4)
                   AA[0:4,4:8] = -(A+2.0*B)
                   AA[4:8,4:8] = np.eye(4)
                   BB[0:4,4:8] = -B 
                   BB[4:8,0:4] = np.eye(4)
                   RES = np.matmul(np.linalg.inv(AA),BB)
               else:    
                   RES = A+B
                   RESI = B
                   for iiter in range(1,Niter):
                       RES = A+np.matmul(B,RES)
                       RESI = np.matmul(B,RESI)
                   EIG_RESI = np.linalg.eigvals(RESI)
                   ABS = np.max(np.abs(EIG_RESI[:]))
                   rmax[j,i,l,it,ig] = ABS
               eigM = np.linalg.eigvals(M)
               EIG = np.linalg.eigvals(RES)
               lmax[j,i,l,it,ig] = np.max(np.abs(EIG[:]))/\
               np.max([np.max(np.abs(np.exp(eigM*dt[j]))),1.0])
    return(lmax,rmax)

#%% Plot routine
def plot_alpha_G_amp(lmax,rmax):
    champ = np.squeeze(lmax)
    champ_new = np.zeros((NT,NG))
    champ_resi = np.squeeze(rmax)
    champ_new_resi = np.zeros((NT,NG))
    for ig in range(NG):
        for it in range(NT):
            try:
                champ_new[it,ig] = np.max(champ[:,:,it,ig])
                champ_new_resi[it,ig] = np.max(champ_resi[:,:,it,ig])
            except:
                champ_new[it,ig] = np.max(champ[:,it,ig])
                champ_new_resi[it,ig] = np.max(champ_resi[:,it,ig])
    couleurs = ['white','whitesmoke',\
                'lightgrey','#666666','#444444','#222222','#000000']      
    level1 = [1.0,1.005,1.02,1.1,1.5,2.0]
    level3 = [0.00001,0.001,0.01,0.05,0.1,0.5,1.0]
    fig, ax = plt.subplots(constrained_layout=True)
    ax3 = ax.twinx()
    C1 = ax.contourf(alpha,G,champ_new.T, levels=level1, colors=couleurs,\
                     extend='both')
    C3 = ax.contour(alpha,G,champ_new_resi.T,levels = level3,colors ='red',\
                    linewidths=(1,),linestyles=['dotted'])
    ax.clabel(C3, colors='red', fontsize=8)  
    C1.cmap.set_over("black")
    C1.cmap.set_under("white")
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'G')
    ax3.set_ylabel(r'$\theta$ (°)')
    plt.yticks((0,1.0/6,2.0/6,0.5,0.5+1.0/6,0.5+2.0/6,1.0),\
               ('0','26.6','45.0','56.3','63.4','68.2','71.6'))
    fig.colorbar(C1,ax=ax3,location='right')
    C1.cmap.set_over("black") 


#%% Call to calculation and plot routines
(lmax,rmax) = ICI(Limp,LSI2TL,k,G,dt,Niter)
plot_alpha_G_amp(lmax,rmax)
