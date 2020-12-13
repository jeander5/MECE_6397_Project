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
#12/12/2020. UPDATE! Update to project statement. Dirchlet Boundary conditions are functions of time.
#Need to modifiy the code
#Modiphy BC with (1-exp(-lamda8t)) imma use phi instead of lamda 

#imports
import math
import numpy as np
import matplotlib.pyplot as plt
from math import sin as sin
from math import cos as cos
from math import exp as exp
#from mpl_toolkits import mplot3d

#Domain of interest
#a_x<x<b_x and a_y<y<b_y
#note not greater than or equal to

a_x = 0 
a_y = 0
b_x = 2*math.pi
b_y = 2*math.pi
#Boundary Conditions, are now also functions of time
#TOP: u(x,y=by,t)=f_a(x)
#BOTTOM: u(x,y=ay,t)=ga(x)
#LEFT: NEUMANN: dudx(x=ax)=0
#defining v here to be consistent and changeable
v=0
#RIGHT: This is a the big one 
#g_a(bx)+(y-a_y)/(b_y-a_y) *[f_a(b_x)-g_a(b_x)]
#choose a value between 0.05 and 0.5.
phi=0.4999
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
#inputs are the x points and the constant in the expression. And now also time and phi.
    func_vals = [x*(x-a)*(x-a) for x in x]
    return func_vals   

def given_g(x, a):
    """returns the f(x) values for the given function"""
#inputs are the x points and the constant in the expression
    func_vals = [cos(x)*(x-a)*(x-a) for x in x]
    return func_vals   

#im gonna call this function RIGHT for now. Is the BC for the right side of the domain

def RIGHT(y, a_x, a_y, b_x, b_y):
    """returns the RIGHT boundary condition values for this problem"""    
#inputs are the boundary points of the domainand the discretized y values
# I will just break this equation up for now     
    uno = cos(b_x)*(b_x-a_x)*(b_x-a_x)
    dos = [(y-a_y)/(b_y-a_y) for y in y]
    tres = b_x*(b_x-a_x)*(b_x-a_x)-uno
    func_vals= [uno + dos*tres for dos in dos]
    return func_vals

#Okay Im gonna leave those equations as is, store them in an initial vector or matrix i dont care
#then make a new function for the time dependant part that gets called.
#that way im not calling 3 functions every time im only calling one
#I will call this function.....BC_Modifier
#But oh that will have to becalled for the Half value matrix and the Sol Next though too, 
def BC_Modifier(t,phi):
    """returns the values that scales the Dirchlet Bbundary conditions. 
    This is just a value between 0 and 1 """    
#inputs are the current time step, not time step mind, and the value phi, which is the constant in the exponential
    func_val=(1-exp(-phi*t))  
    return func_val
   

#Number of discretized points
#Number of internal x points
#N_x=9998
#Number of internal y points
#N_y=9998
N_y= 8
N_x= 8
#just using the same number of points for x and y beccareful with this. i will have to go back and change later...maybe    
#All of my eqautions assume a uniform grid spacing in x and y.
N=N_x
    

#calling the DIF
x, dx = DIF(b_x,N_x)
y, dy = DIF (b_y,N_y)

#lengths, defined here so they wont need to be calculated else where
#I dont think I ever use these tho
len_x=len(x)
len_y=len(y)

#Sol is solution matrix for the nth time step, will be continuosly updated
Sol = np.ones((len_x,len_y))

##Applying Initial condition for internal points
Sol[1:-1,1:-1]=Uo
#Storing Initial "unmodified' Boundary Conditions,
#subscript um for un modified 
#Left, is the neumann
#Sol[:,0]
#Right
right_um=RIGHT(y, a_x, a_y, b_x, b_y)
##Bottom
bottom_um=given_g(x, a_x)
##Top
top_um=given_f(x, a_x)


#And i will do that right now...
#approximating( u(0,k,t=0) with a three point forward approximation)
def TPFF(col1,col2,v,dx):
    N=len(col1)
    col0=np.ones(N)
    for k in range(0,N):
        col0[k]=-3*col1[k]+4*col2[k]+v*2*dx
#        col0[k]=1*k+k*k/N+1.75
    return col0
Sol[1:-1,0]=TPFF(Sol[1:-1,1],Sol[1:-1,2],v,dx)
# still just is zero   


#first im gonna define the Tridiagonal elements, which are the constants from the Crank Nicolson Scheme
#also a delta t, and I geuss a D
dt=0.25
D=1
mu=D*dt/(dx*dx)
#im using mu so I dont have to mispell lambda as lamda
a=(1+mu)
b=-mu/2
c=-mu/2
d=(1-mu)

def ADI(Sol,t):
    """Performs The ADI Method, returns u values at the next time step"""
#This isnt so general right now. It is kinda specific to my problem. Neumann on the left side of domain. 
#Nuemann that scale with time but eventually reach steady state
    
#Note Im keeping the inputs simple right now, so it does infact use variables defined globally     
    # =============================================================================
    # ADI-Method
    
    # =============================================================================
    # i want this function to be self contained...but its gonna have way to many inputs that way
    # It will just have to call these defined globally,
    # =============================================================================
    
    # =============================================================================
    #Defining Half value matrix, for u(t=n+1/2), and Sol_next for u(t=n+1)
    HVM=np.ones((len_x,len_y))
    Sol_next=np.ones((len_x,len_y))
    #Now applying boundary conditions to the nth, nth+1/2, and nth plus 1 time step.
    mod1=BC_Modifier(t,phi)
    mod2=BC_Modifier(t+dt/2,phi)
    mod3=BC_Modifier(t+dt,phi)
    #Left, is the neumann
    #Right
    Sol[:,-1]=[x*mod1 for x in right_um]
    HVM[:,-1]=[x*mod2 for x in right_um]
    Sol_next[:,-1]=[x*mod3 for x in right_um]
    ###Bottom
    Sol[0,:]=[x*mod1 for x in bottom_um]
    HVM[0,:]=[x*mod2 for x in bottom_um]
    Sol_next[0,:]=[x*mod3 for x in bottom_um]
    ###Top
    Sol[-1,:]=[x*mod1 for x in top_um]
    HVM[-1,:]=[x*mod2 for x in top_um]
    Sol_next[-1,:]=[x*mod3 for x in top_um]
    
#this is the ghost node term that appears in some equations. It is calculate here 
#it doesnt have to be repeatedly calculated. For my problem its just zero.    
    GNT=b*2*dx*v
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
    
    for k in range(1,N+1):
    ##for now I will just redefine inside for troubleshooting purposes    
#        rhs_a = np.ones(N+1)
    ##first eq different    
        rhs_a[0] = -b*Sol[k-1, 0] + d*Sol[k, 0] - c*Sol[k+1, 0] + GNT
    #    b*v*2*dx from ghost node, even if it is zero
        ##middle eqs
        for j in range(1, N):
            rhs_a[j] = -b*Sol[k-1, j] + d*Sol[k, j] - c*Sol[k+1, j]
    ##last eq different
        rhs_a[-1] = -b*Sol[k-1,-2] + d*Sol[k,-2] - c*Sol[k+1,-2] - c*HVM[k,-1]   
        HVM[k,0:-1]=thomas_alg_func(av,bv,cv,rhs_a)
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
    #first all the u(0,k) for k=1,2...N)      
    #these are a little different because of the ghost node.
    #first equation different
    rhs_b[0]= -b*HVM[1, 1]- GNT + d*HVM[1, 0] - c*HVM[1, 1] -b*Sol_next[0,0]
    #middle eqs
    for k in range(2, N):
        rhs_b[k-1] = -b*HVM[k, 1] - GNT + d*HVM[k, 0] - c*HVM[k, 1]
    #last equation different
        rhs_b[-1]= -b*HVM[N, 1] - GNT + d*HVM[N, 0] - c*HVM[N, 1] - c*Sol_next[N+1,0]
    Sol_next[1:-1,0]=thomas_alg_func(av_b,bv_b,cv_b,rhs_b)
    
    #now for the next ones, u(j,k) for for j=1,2...N and k=1,2...N)
    for j in range(1,N+1):
    #just redifining rhs_every time ever for now while im still trouble shooting   
#        rhs_b=np.ones((N))   
    #first equation different
        rhs_b[0]= -b*HVM[1, j-1] + d*HVM[1, j] - c*HVM[1, j+1] -b*Sol_next[0, j]
    #middle eqs
        for k in range(2, N):
            rhs_b[k-1] = -b*HVM[k, j-1] + d*HVM[k, j] - c*HVM[k, j+1]
    #last equation different
        rhs_b[-1]= -b*HVM[N, j-1] + d*HVM[N, j] - c*HVM[N, j+1]-c*Sol_next[N+1, j]
        Sol_next[1:-1,j]=thomas_alg_func(av_b,bv_b,cv_b,rhs_b)
    return (Sol_next)
# =============================================================================
# I wanna bring what ever I can outside that function to make it run faster
# And Im really okay with having the function use the some global variables.    
# Or should I just buck up adn make the function have many inputs.
# I could but many inputs in a vector and unpack the vector, so its not so messy to call it.
# =============================================================================


# =============================================================================
# Advancing solution forward in time
# =============================================================================
#initial time=0
t=0
#calclulating solution
for m in range(51):
    Sol_next = ADI(Sol,t)
#check error stuff, grid convergence ect
#if not steady Sol=Sol_next
    Sol = Sol_next     
#  advance time
    t = t + dt
Sol_next = ADI(Sol,t)

#for the errors we will not check the boundary conditions 
#only the left nodes, that I am solving for. But those are actually outside my domain
#so even thought I am solving for them We will just check points inside the domain    
#Generic variable G1, used in some error calculations
#Just keeps the code a little cleaner
G1=abs(Sol_next[1:-1,1:-1]-Sol[1:-1,1:-1])
L1_error=(1/(N*N))*np.sum(G1)
#I brought the 1/N^2 out of the summation, all my equations assume dx=dy.
L2_error=(1/N)*math.sqrt(np.sum(G1*G1))
Linf_error=G1.max()
#
#Okay to do the grid convergence Log2 equation I really need to store three solutions....
#I really dont like doing the GCS in a while loop I like just like doing it manually really.
#And for the L1 L2 Linf it is really about grid spacing and not time step...
#also prosperetti in the instructions uses relative errror for the norms it shouldnt matter though




















#m=0
#m=m+1
#t=dt*m
#Sol_next=ADI(Sol,t)
#m=m+1
#t=dt*m
#Sol_next=ADI(Sol,t)
    
# =============================================================================
# Plotting
# =============================================================================
##bottom
fig5, ax5 = plt.subplots()
plt.grid(1)    
plt.plot(y,Sol_next[0,:],'-b') 
plt.plot(y,Sol_next[1,:],':r')
plt.xlabel('x')
plt.ylabel('u(x, y)')
ax5.legend(['u(x, a_y)','u(x, y_1)'])
ax5.title.set_text('BOTTOM At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
                   %(round(t,4),round(dx,4),round(dt,4),round(mu,4)))  
#top
fig6, ax6 = plt.subplots()
plt.grid(1)    
plt.plot(y,Sol_next[-1,:],'-b') 
plt.plot(y,Sol_next[-2,:],':r')
plt.xlabel('x')
plt.ylabel('u(x, y)')
ax6.legend(['u(x, b_y)','u(x, y_N)'])
ax6.title.set_text('TOP: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
                   %(round(t,4), round(dx,4),round(dt,4),round(mu,4)))  
#left
fig7, ax7 = plt.subplots()
plt.grid(1)
plt.plot(y,Sol_next[:,0],'-b') 
plt.plot(y,Sol_next[:,1],':r')
plt.xlabel('y')
plt.ylabel('u(x, y)')
ax7.legend(['u(a_x, y)','u(x_1, y)'])
ax7.title.set_text('LEFT: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
                   %(round(t,4), round(dx,4),round(dt,4),round(mu,4)))  
#right
fig8, ax8 = plt.subplots()
plt.grid(1)    
plt.plot(y,Sol_next[:,-1],'-b') 
plt.plot(y,Sol_next[:,-2],':r')     
plt.xlabel('y')
plt.ylabel('u(x, y)')
ax8.legend(['u(b_x, y)','u(x_N, y)'])  
ax8.title.set_text('RIGHT: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
                   %(round(t,4), round(dx,4),round(dt,4),round(mu,4)))

#3D GRAPH
#from mpl_toolkits import mplot3d
fig10 = plt.figure()
ax10 = plt.axes(projection='3d')
BIGX, BIGY = np.meshgrid(x, y)    
from matplotlib import cm    
surf = ax10.plot_surface(BIGX, BIGY, Sol_next, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
#fig2.colorbar(surf, shrink=0.75, aspect=5)
ax10.set_title('u(x,y,t=%s s) \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s '%(round(dt*t,4),round(dx,4),round(dt,4),round(mu,4)))
ax10.set_xlabel('x')
ax10.set_ylabel('y')
ax10.set_zlabel('u')    


#I still get the overshoot, the dip the spike ect.
#I really dont think it is a coding error.
#I think the its just an error from having too big a time step, the scheme is still stable and will reach a steady state solution
#Its going from zero to like 20 in one time step
#If it is a coding error I would be very interested in knowing where and what it was
#And why for small mu it does not occur.
#what that little dip will do is make my error bigger and make me run more time steps that i really need to reach steady state
#whart if I try a big mu but a small time? Same
#It must be an error somewhere I just cant see it.
#It is interesting to compare the first time step to the second
#oh shoot wait I just fixed it. When I was just running it I was applying the boundary conditions a step ahead.
#ok nice
#so it still is an accuracy kinda in the scheme with too big a time step I think
#where did my 3d plot go???/
#And I should really make a subplot of those left right bottom top ect
#I like having the individuals tho, the subplots are too small