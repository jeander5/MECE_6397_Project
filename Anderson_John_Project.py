# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:01:42 2020
@author: johna
"""


# MECE 6397, SciComp, Grad Student Project

# 2-D Diffusion Problem 
#Carryout integration intill steady state solution is reached
#used ghost node method for nuemann boundary conditions
#https://github.com/jeander5/MECE_6397_Project

#imports
import math
import numpy as np
import matplotlib.pyplot as plt
from math import sin as sin
from math import cos as cos
#from mpl_toolkits import mplot3d

#Domain of interest
#a_x<x<b_x and a_y<y<b_y
#note not greater than or equal to

a_x = 0 
a_y = 0
b_x = 2*math.pi
b_y = 2*math.pi

#Boundary Conditions
#TOP: u(x,y=by)=f_a(x)
#BOTTOM: u(x,y=ay)=ga(x)
#LEFT: NEUMANN: dudx(x=ax)=0
#defining v here to be consistent and changeable
v=0
#RIGHT: This is a the big one 
#g_a(bx) +(y-a_y)/(b_y-a_y) *[f_a(b_x)-g_a(b_x)]

#Initial conditions, are just zero for all points, INSIDE the boundary
#U(x,y,t)=0
#I will still define this.
Uo=0


#Defining Functions

def thomas_alg_func(a,b,c,f):
    """solves tridiagonal matrix"""
#inputs are vectors containing the tridiagonal elements and right hand side
    N=len(a)
#storage vectors    
    u_appx = [0]*N
    alpha = [0]*N
    g = [0]*N
#Following the pseudocode
#Zeroth element of this list corresponds to the first subscript in Thomas Algorithm
    alpha[0] = a[0]
    g[0] = f[0]
    for j in range(1, N):
        alpha[j] = a[j]-(b[j]/alpha[j-1])*c[j-1]
        g[j] = f[j]-(b[j]/alpha[j-1])*g[j-1]
    u_appx[N-1] = g[N-1]/alpha[N-1]
    for j in range(1, N):
        u_appx[-1-j] = (g[-1-j]-c[-1-j]*u_appx[-j])/alpha[-1-j]
    return u_appx    

#lets bring in that discretize the interval function
def DIF(L ,N):
    """discretizes an interval with N interior points"""
#Inputs are interval length number of interior points  
#Returns the discretize domain and the interval spacing
#Includes endpoints    
    h = L/(N+1)
    x = np.linspace(0, L, N+2)
    return(x, h)

#The given functions f and g
#these functions control the boundary conditions
#vector inputs for these functions must include the endpoints

def given_f(x, a):
    """returns the f(x) values """
#inputs are the x points and the constant in the expression
    func_vals = [x*(x-a)*(x-a) for x in x]
    return func_vals   

def given_g(x,a):
    """returns the f(x) values for the given function"""
#inputs are the x points and the constant in the expression
    func_vals = [cos(x)*(x-a)*(x-a) for x in x]
    return func_vals   

#im gonna call this function RIGHT for now

def RIGHT(y, a_x, a_y, b_x, b_y):
    """returns the RIGHT Boundary condition values for this problem"""    
#inputs are the boundary points of the domainand the discretized y values
# I will just break this equation up for now     
    uno = cos(b_x)*(b_x-a_x)*(b_x-a_x)
    dos = [(y-a_y)/(b_y-a_y) for y in y]
    tres = b_x*(b_x-a_x)*(b_x-a_x)-uno
    func_vals= [uno + dos*tres for dos in dos]
    return func_vals  

#Number of discretized points
#Number of internal x points
N_x=254
#Number of internal y points
N_y=254
#just using the same number of points for x and y beccareful with this. i will have to go back and change later...maybe    
N=N_x
    
#calling the DIF
x, dx = DIF(b_x,N_x)
y, dy = DIF (b_y,N_y)

#lengths, defined here so they wont need to be calculated else where
len_x=len(x)
len_y=len(y)

#Sol is solution matrix for the nth time step
Sol = np.ones((len_x,len_y))

#Applying Boundary Conditions
#Left, is the neumann
#Sol[:,0]
#Right
Sol[:,-1]=RIGHT(y, a_x, a_y, b_x, b_y)
##Bottom
Sol[0,:]=given_g(x, a_x)
##Top
Sol[-1,:]=given_f(x, a_x)
##Applying Initial condiots
Sol[1:-1,1:-1]=Uo

#And i will do that right now...
#approximating( u^n_0,k) with a three point forward approximation)
#I will make a function for this and keep it here for now.
def TPFF(col1,col2,v,dx):
    N=len(col1)
    col0=np.ones(N)
    for k in range(0,N):
        col0[k]=-3*col1[k]+4*col2[k]+v*2*dx
#        col0[k]=1*k+k*k/N+1.75
    return col0

##now I will define Sol_next, using copy()
Sol_next=Sol.copy()
#im doing this before apply the 3 point forward method.
Sol[1:-1,0]=TPFF(Sol[1:-1,1],Sol[1:-1,2],v,dx)      
# still just is zero   
 

# =============================================================================
#  Neumann Boundary Condtion, Left, x=ax=0, GHOST NODE METHOD
# =============================================================================

#first im gonna define the Tridiagonal elements, which are the constants from the Crank Nicolson Scheme
#also a delta t, and I geuss a D
dt=.1
D=1
#and dx=dy
mu=D*dt/(dx*dx)
#im using mu so I dont have to mispell lambda as lamda
a=(1+mu)
b=-mu/2
c=-mu/2
d=(1-mu)

# =============================================================================
# ADI-Method
# =============================================================================
#Half value matrix, for u(t=n+1/2)
HVM=Sol.copy()
#its quicker just to make a copy rather than define a blank matrix and reapply the boundary conditions to the half value matrix
# =============================================================================
# #Step "A":The t=n to t=n+ 1/2 step
# =============================================================================
rhs_a=np.ones(N+1)
#rhs for step a is (N+1) because of the ghost node
#Pre thomas algorithm set up
av= [a]*(N+1)
bv= [b]*(N+1)
cv= [c]*(N+1)
#different first input because of ghost node
cv[0]=b+c 
# =============================================================================
# =============================================================================
# The ERROR IS ACTUALLY IN HERE!
# THE PENULTIMATE ROW IS JUST WRONG!!!!!!!! FIX IT
# =============================================================================
# =============================================================================
#need to do this N times, for k=1,2..N
for k in range(1,N+1):
##for now I will just redefine inside for troubleshooting purposes    
    rhs_a = np.ones(N+1)
##first eq different    
    rhs_a[0] = -b*Sol[k-1, 0] + d*Sol[k, 0] - c*Sol[k+1, 0] + b*v*2*dx
#    b*v*2*dx from ghost node, even if it is zero
    ##middle eqs
    for j in range(1, N):
##I think ive checked this already
#    for j in range(1, N+1):
        rhs_a[j] = -b*Sol[k-1, j] + d*Sol[k, j] - c*Sol[k+1, j]
##last eq different
    rhs_a[-1] = -b*Sol[k-1,-2] + d*Sol[k,-2] - c*Sol[k+1,-2] - c*HVM[k,-1]   
    HVM[k,0:-1]=thomas_alg_func(av,bv,cv,rhs_a)
#right now, this all the n+1/2 values,
 
# =============================================================================
# #Step "B" the t=n+1/2 to t=n+1
# =============================================================================
#must solve one column at a time to preserve tridiagonal structure
#I would have to modify every Nth element of the b and c vectors, is fine for now
#okay rhs b still is N long, I just need to do the steps N+1 one times    
rhs_b=np.ones(N)
av_b= [a]*(N)
bv_b= [b]*(N)
cv_b= [c]*(N)    
# =============================================================================
# I need to focus on this section here, for big mu values. THERE MUST BE AN ERROR
# =============================================================================

#first all the u(0,k) for k=1,2...N)      
#these are a little different because of the ghost node.
#first equation different
rhs_b[0]= -b*(HVM[1, 1]-2*dx*v) + d*HVM[1, 0] - c*HVM[1, 1] -b*Sol_next[0,0]
#middle eqs
for k in range(2, N):
    rhs_b[k-1] = -b*(HVM[k, 1]-2*dx*v) + d*HVM[k, 0] - c*HVM[k, 1]
#last equation different
    rhs_b[-1]= -b*(HVM[N, 1]-2*dx*v) + d*HVM[N, 0] - c*HVM[N, 1] - c*Sol_next[N+1,0]
Sol_next[1:-1,0]=thomas_alg_func(av_b,bv_b,cv_b,rhs_b)


#now for the next ones, u(j,k) for for j=1,2...N and k=1,2...N)
for j in range(1,N+1):
#reset k to one at beginnig of this loop
    k=1
#just redifining rhs_every time ever for now while im still trouble shooting   
    rhs_b=np.ones((N))   
#first equation different
    rhs_b[0]= -b*HVM[k, j-1] + d*HVM[k, j] - c*HVM[k, j+1] -b*Sol_next[k-1, j]
#middle eqs
    for k in range(2, N):
        rhs_b[k-1] = -b*HVM[k, j-1] + d*HVM[k, j] - c*HVM[k, j+1]
#last equation different
    rhs_b[-1]= -b*HVM[N, j-1] + d*HVM[N, j] - c*HVM[N, j+1]-c*Sol_next[N+1, j]
    Sol_next[1:-1,j]=thomas_alg_func(av_b,bv_b,cv_b,rhs_b)
  
#Okay, why do I have this part B in two different steps? that could be a sign that something is wrong    

#def ADI(Sol):
#    """Quick and dirty ADI function right now so I can advance the solution forward a few time steps"""   
#    # =============================================================================
#    # ADI-Method
#    # =============================================================================
#    #Half value matrix, for u(t=n+1/2)
#    HVM=Sol.copy()
#    #its quicker just to make a copy rather tahn define a blank matrix and reapply the boundary conditions to the half value matrix
#    # =============================================================================
#    # #Step "A":The t=n to t=n+ 1/2 step
#    # =============================================================================
#    rhs_a=np.ones(N+1)
#    #rhs for step a is (N+1) because of the ghost node
#    #Pre thomas algorithm set up
#    av= [a]*(N+1)
#    bv= [b]*(N+1)
#    cv= [c]*(N+1)
#    #different first input because of ghost node
#    cv[0]=b+c 
#    
#    #need to do this N times, for k=1,2..N
#    for k in range(1,N+1):
#    ##for now I will just redefine inside for troubleshooting purposes    
#        rhs_a = np.ones(N+1)
#    ##first eq different    
#        rhs_a[0] = -b*Sol[k-1, 0] + d*Sol[k, 0] - c*Sol[k+1, 0] + b*v*2*dx
#    #    b*v*2*dx from ghost node, even if it is zero
#        ##middle eqs
#        for j in range(1, N):
#    ##I think ive checked this already
#    #    for j in range(1, N+1):
#            rhs_a[j] = -b*Sol[k-1, j] + d*Sol[k, j] - c*Sol[k+1, j]
#    ##last eq different
#        rhs_a[-1] = -b*Sol[k-1,-2] + d*Sol[k,-2] - c*Sol[k+1,-2] - c*HVM[k,-1]   
#        HVM[k,0:-1]=thomas_alg_func(av,bv,cv,rhs_a)
#    #right now, this all the n+1/2 values,
#     
#    # =============================================================================
#    # #Step "B" the t=n+1/2 to t=n+1
#    # =============================================================================
#    #!must solve one column at a time to preserve tridiagonal structure
#    #I would have to modify every Nth element of the b and c vectors, is fine for now
#    #okay rhs b still is N long, I just need to do the steps N+1 one times    
#    rhs_b=np.ones(N)
#    av_b= [a]*(N)
#    bv_b= [b]*(N)
#    cv_b= [c]*(N)    
#    # =============================================================================
#    # I need to focus on this section here, for big mu values. THERE MUST BE AN ERROR
#    # =============================================================================
#    #DO I really have to have two different parts here because of the ghost node? Yes I think so
#    #lets let v=5 and see what happens
#    #i really think I am applying the GBNT correctly. changing v inverts the final paraobla I get. 
#    #though there still may be a a sign error? because if I do a positive v, as in like say
#    #heat into the system my parabola inverts. where it seems like that should make my parabola get taller
#    #oh never mind its just the sign convention on q probably...no still seems wrong. idk move on 
#    
#    
#    #first all the u(0,k) for k=1,2...N)      
#    #these are a little different because of the ghost node.??
#    #first equation different
#    rhs_b[0]= -b*(HVM[1, 1]-2*dx*v) + d*HVM[1, 0] - c*HVM[1, 1] -b*Sol_next[0,0]
#    #middle eqs
#    for k in range(2, N):
#        rhs_b[k-1] = -b*(HVM[k, 1]-2*dx*v) + d*HVM[k, 0] - c*HVM[k, 1]
#    #last equation different
#        rhs_b[-1]= -b*(HVM[N, 1]-2*dx*v) + d*HVM[N, 0] - c*HVM[N, 1] - c*Sol_next[N+1,0]
#    Sol_next[1:-1,0]=thomas_alg_func(av_b,bv_b,cv_b,rhs_b)
#    
#    
#    #now for the next ones, u(j,k) for for j=1,2...N and k=1,2...N)
#    for j in range(1,N+1):
#    #reset k to one at beginnig of this loop
#        k=1
#    #just redifining rhs_every time ever for now while im still trouble shooting   
#        rhs_b=np.ones((N))   
#    #first equation different
#        rhs_b[0]= -b*HVM[k, j-1] + d*HVM[k, j] - c*HVM[k, j+1] -b*Sol_next[k-1, j]
#    #middle eqs
#        for k in range(2, N):
#            rhs_b[k-1] = -b*HVM[k, j-1] + d*HVM[k, j] - c*HVM[k, j+1]
#    #last equation different
#        rhs_b[-1]= -b*HVM[N, j-1] + d*HVM[N, j] - c*HVM[N, j+1]-c*Sol_next[N+1, j]
#        Sol_next[1:-1,j]=thomas_alg_func(av_b,bv_b,cv_b,rhs_b)
#    return Sol_next
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
##first Step, just to compare and understand
#FS=Sol_next.copy()
#for t in range (501):
#    Sol_next=ADI(Sol_next)





























t=1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
#plotting 
#bottom
#fig1, ax1 = plt.subplots()
#plt.grid(1)    
#plt.plot(y,FS[0,:],'-b') 
#plt.plot(y,FS[1,:],':r')
#plt.xlabel('x')
#plt.ylabel('u(x, y)')
#ax1.legend(['u(x, a_y)','u(x, y_1)'])
#ax1.title.set_text('At first time step \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(dx,4),round(dt,4),round(mu,4)))  
##top
#fig2, ax2 = plt.subplots()
#plt.grid(1)    
#plt.plot(y,FS[-1,:],'-b') 
#plt.plot(y,FS[-2,:],':r')
#plt.xlabel('x')
#plt.ylabel('u(x, y)')
#ax2.legend(['u(x, b_y)','u(x, y_N)'])
#ax2.title.set_text('At first time step \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(dx,4),round(dt,4),round(mu,4)))  
##left
#fig3, ax3 = plt.subplots()
#plt.grid(1)
#plt.plot(y,FS[:,0],'-b') 
#plt.plot(y,FS[:,1],':r')
#plt.xlabel('y')
#plt.ylabel('u(x, y)')
#ax3.legend(['u(a_x, y)','u(x_1, y)'])
#ax3.title.set_text('At first time step \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(dx,4),round(dt,4),round(mu,4)))  
##right
#fig4, ax4 = plt.subplots()
#plt.grid(1)    
#plt.plot(y,FS[:,-1],'-b') 
#plt.plot(y,FS[:,-2],':r')     
#plt.xlabel('y')
#plt.ylabel('u(x, y')
#ax3.legend(['u(b_x, y)','u(x_N, y)'])  
#ax3.title.set_text('At first time step \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(dx,4),round(dt,4),round(mu,4)))  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#fig, ax = plt.subplots()
#plt.grid(1)
###plt.plot(y,Sol[4,:],'-.c')
###plt.plot(y,HVM[4,:],':r')
###plt.plot(y,Sol_next[4,:],'-b')
##bottom
#fig1, ax1 = plt.subplots()
#plt.grid(1)    
#plt.plot(y,Sol_next[0,:],'-b') 
#plt.plot(y,Sol_next[1,:],':r')
##top
#fig2, ax2 = plt.subplots()
#plt.grid(1)    
#plt.plot(y,Sol_next[-1,:],'-b') 
#plt.plot(y,Sol_next[-2,:],':r')
##left
#fig3, ax3 = plt.subplots()
#plt.grid(1)
#plt.plot(y,Sol_next[:,0],'-b') 
#plt.plot(y,Sol_next[:,1],':r')
##right
#fig4, ax4 = plt.subplots()
#plt.grid(1)    
#plt.plot(y,Sol_next[:,-1],'-b') 
#plt.plot(y,Sol_next[:,-2],':r')           
###    right now I think the issue is in the last equation, the u(N,k) equation
##ax.legend(['u(0,y,t^n)','u(x_1,y,t^n+1)'])
#plt.xlabel('x')
#plt.ylabel('u(x,y,t)')
#ax.title.set_text('Bottom boundary condition and u(x,y_1,t^n+1) \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(dx,4),round(dt,4),round(mu,4)))  
#ax.title.set_text('Right boundary condition and u(x_N,y,t^n+1) \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(dx,4),round(dt,4),round(mu,4)))
#ax.title.set_text('u(0,y,t^n) and u(x_1,y,t^n+1) \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(dx,4),round(dt,4),round(mu,4)))    
    
    
 
#3D GRAPH
from mpl_toolkits import mplot3d
fig10 = plt.figure()
ax10 = plt.axes(projection='3d')
BIGX, BIGY = np.meshgrid(x, y)    
from matplotlib import cm    
surf = ax10.plot_surface(BIGX, BIGY, Sol_next, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
#fig10.colorbar(surf, shrink=0.75, aspect=5)
ax10.set_title('u(x,y,t=%s s) \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s '%(round(dt*t,4),round(dx,4),round(dt,4),round(mu,4)))
ax10.set_xlabel('x')
ax10.set_ylabel('y')
ax10.set_zlabel('u')    
#
#
#
##more plotting
t=1
##bottom
fig5, ax5 = plt.subplots()
plt.grid(1)    
plt.plot(y,Sol_next[0,:],'-b') 
plt.plot(y,Sol_next[1,:],':r')
plt.xlabel('x')
plt.ylabel('u(x, y)')
ax5.legend(['u(x, a_y)','u(x, y_1)'])
ax5.title.set_text('BOTTOM At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
                   %(round(dt*t,4),round(dx,4),round(dt,4),round(mu,4)))  
#top
fig6, ax6 = plt.subplots()
plt.grid(1)    
plt.plot(y,Sol_next[-1,:],'-b') 
plt.plot(y,Sol_next[-2,:],':r')
plt.xlabel('x')
plt.ylabel('u(x, y)')
ax6.legend(['u(x, b_y)','u(x, y_N)'])
ax6.title.set_text('TOP: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
                   %(round(dt*t,4), round(dx,4),round(dt,4),round(mu,4)))  
##left
#fig7, ax7 = plt.subplots()
#plt.grid(1)
#plt.plot(y,Sol_next[:,0],'-b') 
#plt.plot(y,Sol_next[:,1],':r')
#plt.xlabel('y')
#plt.ylabel('u(x, y)')
#ax7.legend(['u(a_x, y)','u(x_1, y)'])
#ax7.title.set_text('LEFT: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(dt*t,4), round(dx,4),round(dt,4),round(mu,4)))  
#right
fig8, ax8 = plt.subplots()
plt.grid(1)    
plt.plot(y,Sol_next[:,-1],'-b') 
plt.plot(y,Sol_next[:,-2],':r')     
plt.xlabel('y')
plt.ylabel('u(x, y)')
ax8.legend(['u(b_x, y)','u(x_N, y)'])  
ax8.title.set_text('RIGHT: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
                   %(round(dt*t,4), round(dx,4),round(dt,4),round(mu,4)))
