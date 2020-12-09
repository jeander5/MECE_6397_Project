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
N_x=8    
#Number of internal y points
N_y=8
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
#Bottom
Sol[0,:]=given_g(x, a_x)
#Top
Sol[-1,:]=given_f(x, a_x)
#Applying Initial condiots
Sol[1:-1,1:-1]=Uo

#And i will do that right now...
#approximating( u^n_0,k) with a three point forward approximation)
#I will make a function for this and keep it here for now.
def TPFF(col1,col2,v,dx):
    N=len(col1)
    col0=np.ones(N)
    for k in range(0,N):
        col0[k]=-3*col1[k]+4*col2[k]+v*2*dx
    return col0

##now I will define Sol_next, using copy()
Sol_next=Sol.copy()
#im doing this before apply the 3 point forward method.
Sol[1:-1,0]=TPFF(Sol[1:-1,1],Sol[1:-1,2],v,dx)      
    
 

# =============================================================================
#  Neumann Boundary Condtion, Left, x=ax=0, GHOST NODE METHOD
# =============================================================================

#first im gonna define the Tridiagonal elements, which are the constants from the Crank Nicolson Scheme

#also a delta t, and I geuss a D
dt=0.1
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
#Wait I should really change this HVM to be the same size as Sol, with the boundary conditions already applied
HVM=Sol.copy()
#its quicker just to make a copy rather tahn define a blank matrix and reapply the boundary conditions to the half value matrix
#Step "A":The t=n to t=n+ 1/2 step.
rhs_a=np.ones(N+1)
#rhs for step a is (N+1) because of the ghost node


#Pre thomas algorithm set upo
#okay, the a,b,c and vectors, (N+1) because of ghost node
#its fine that these are lists and not real numpy arrays
#these dont change
av= [a]*(N+1)
bv= [b]*(N+1)
#different first input because of ghost node
bv[0]=b+c
cv= [c]*(N+1)

for m in range(0,N):

#filling in the rhs vector   
#Really i can just writre over the values as we go and define outside the loop  
#I swiched to s and m rather than j adn k because i was getting confused for some reason before
#Now I wanna switcfh back
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  keep for now, I dont like this any more
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#I wanna use j,k for sol and HVM, 
#and then do things like rhs[j-1} becasue rhs is only internal points     
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#but for now I will just redefine    
    rhs_a = np.ones(N+1)
#first eq different    
    rhs_a[0] = -b*Sol[m, 0] + d*Sol[m+1, 0] - c*Sol[m+2, 0]
#middle eqs
    for s in range(1, N):
        rhs_a[s] = -b*Sol[m, s] + d*Sol[m+1, s] - c*Sol[m+2, s]
#last eq different
    rhs_a[-1] = -b*Sol[m,-2] + d*Sol[m+1,-2] - c*Sol[m+2,-2] - c*Sol[1+m,-1]     
    HVM[m+1,0:-1]=thomas_alg_func(av,bv,cv,rhs_a)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#right now, this all the n+1/2 values,
#the N+1 half point values can be gotten from the Sol matrix, they are right endpoints
#same with bottom and top 

#now, rhs_b is composed from these t= n+1/2 (half vals) 
#I need to pay close attention to the order here. Which is why i stored them in that matrix.
#I am filling up the t=n+1 matrix column by column
    
#Step "B" the t=n+1/2 to t=n+1
#okay rhs b still is N long, I just need to do the steps N+1 one times
#which means I need to redine my or make new thomas algorith vectors. I will just make new ones.
av_b= [a]*(N)
bv_b= [b]*(N)
#different first input because of ghost node
cv_b= [c]*(N)    
    
#I can either to it one column at a time or do them all at once.
#    and I am solving for the u(x=0,k) points as well, so (N+1) by N for I make one long vector
#rhs_b=np.ones((N+1)*N)    
#lets just do one row right now
rhs_b=np.ones(N)
#first all the u(0,k) for k=1,2...N) really its Ny, I need to be careful here I was just using N above       
#these are a little different because of the ghost node.
#first equation different
rhs_b[0]= -(b + c)*HVM[1, 1] + d*HVM[1, 0]  -b*Sol_next[0,0]
#middle eqs
for k in range(2, N):
    rhs_b[k-1] = -(b + c)*HVM[k, 1] + d*HVM[k, 0]
#last equation different
    rhs_b[-1]= -(b + c)*HVM[N, 1] + d*HVM[N, 0] - c*Sol_next[N+1,0]
Sol_next[1:-1,0]=thomas_alg_func(av_b,bv_b,cv_b,rhs_b)
    
#before I changed the indexes to m and s, because I didnt like seeing A[k,j] for example
#but now I find it more confusing
#Im gonna switch back to j and k, even though k will will come before j

#now for the next ones, In a loop 
rhs_b=np.ones((N+1))
for j in range(1,N+1):      
#first equation different
    rhs_b[0]= -b*HVM[k, j-1] + d*HVM[k, j] - c*HVM[k, j+1] -b*Sol_next[k-1, j]
#middle eqs
    for k in range(2, N+1):
        rhs_b[k-1] = -b*HVM[k, j-1] + d*HVM[k, j] - c*HVM[k, j+1]
#last equation different
    rhs_b[-1]= -b*HVM[k, j-1] + d*HVM[k, j] - c*HVM[k, j+1]-c*Sol_next[k-1, j]
    Sol_next[1:-1,j]=thomas_alg_func(av_b,bv_b,cv_b,rhs_b)