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
from scipy import interpolate
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
v = 0
#RIGHT: This is a the big one 
#g_a(bx)+(y-a_y)/(b_y-a_y) *[f_a(b_x)-g_a(b_x)]
#choose a value between 0.05 and 0.5.
phi = 0.4999
#Initial conditions, are just zero for all points, INSIDE the boundary
#U(x,y,t)=0
#I will still define this.
Uo = 0
#defining delta t right here
dt = 0.1

#Defining Functions
def thomas_alg_func(a, b, c, f):
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

# discretize the interval function
def DIF(L ,N):
    """discretizes an interval with N interior points"""
#Inputs are interval length number of interior points  
#Returns the discretize domain and the interval spacing
#Includes endpoints    
    h = L/(N + 1)
    x = np.linspace(0, L, N + 2)
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
    tres = b_x*(b_x - a_x)*(b_x - a_x) - uno
    func_vals = [uno + dos*tres for dos in dos]
    return func_vals
#updated problem statement...............
#Okay Im gonna leave those equations as is, store them in an initial vector or matrix i dont care
#then make a new function for the time dependant part that gets called.
#that way im not calling 3 functions every time im only calling one
#I will call this function.....BC_Modifier
#But oh that will have to be called for the Half value matrix and the Sol Next though too, 
def BC_Modifier(t, phi):
    """returns the values that scales the Dirchlet Boundary conditions. 
    This is just a value between 0 and 1 """    
#inputs are the current time step, not time step mind, and the value phi, which is the constant in the exponential
    func_val = (1 - exp(-phi*t))  
    return func_val
#approximating( u(0,k,t=0) with a three point forward approximation)
def TPFF(col1, col2, v, dx):
    N = len(col1)
    col0 = np.ones(N)
    for k in range(0, N):
        col0[k] = -3*col1[k] + 4*col2[k] + v*2*dx
#        col0[k]=1*k+k*k/N+1.75
    return col0
#Sol[1:-1,0]=TPFF(Sol[1:-1,1],Sol[1:-1,2],v,dx)

#error functions
def L1_norm(u, u_ref):
    G = abs((u - u_ref)/u)
    N = len(u)
    L1_error = (1/(N*N))*np.sum(G)
    return L1_error

def L2_norm(u, u_ref):
##I brought the 1/N^2 out of the summation, all my equations assume dx=dy.
    G = abs((u - u_ref)/u)
    N = len(u)
    ##I brought the 1/N^2 out of the sqrt, all my equations assume dx=dy.
    L2_error = (1/N)*math.sqrt(np.sum(G*G))
    return L2_error


# =============================================================================
# Im just gonna make a set up function
# =============================================================================
def Set_up(N, dt):
#everything  assume the same dx and dy so i will eliminate some duplicate things that were in previous version
#this also will call a few global variable
#this would just need to be modified a bit for different problems     
    x, dx = DIF(b_x, N)
#    x=y and dx=dy
#TP, Total points defined here so they wont need to be calculated else where
    TP = N + 2

#Sol is solution matrix for the nth time step, will be continuosly updated
    Sol = np.ones((TP, TP))

##Applying Initial condition for internal points
    Sol[1:-1, 1:-1] = Uo
#Storing Initial "unmodified' Boundary Conditions, subscript um for un modified 
#Left, is the neumann
#left initial using three point forward method
    Sol[1:-1, 0] = TPFF(Sol[1:-1, 1], Sol[1:-1, 2], v, dx)    
#Right
    right_um = RIGHT(x, a_x, a_y, b_x, b_y)
##Bottom
    bottom_um = given_g(x, a_x)
##Top
    top_um = given_f(x, a_x)
#time for bc to reach steady state dont really need that
#    t_bc=round(4.605/phi)
#    steps_bc=round(t_bc/dt)    
    D = 1
    mu = D*dt/(dx*dx)
#tridiagonal constants from the scheme    
    a = 1 + mu
    b = -mu/2
    c = -mu/2
    d = 1 - mu
    return (Sol, x, dx, right_um, bottom_um, top_um, mu, a, b, c, d, TP)

def ADI(Sol, x, dx, right_um, bottom_um, top_um, mu, a, b, c, d, TP, q):
    """Performs The ADI Method, returns u values at the next time step"""
#now includes an x input, and a dx input, was needed for the grid convergence
#This isnt so general right now. It is kinda specific to my problem. Neumann on the left side of domain. 
#Dirchlet that scale with time but eventually reach steady state
     
    # =============================================================================
    # ADI-Method
    
    # =============================================================================
    # i want this function to be self contained...but its gonna have way to many inputs that way
    # It will just have to call these defined globally,
    # =============================================================================
    
    # =============================================================================
    #Defining Half value matrix, for u(t=n+1/2), and Sol_next for u(t=n+1)
    N = TP - 2
    HVM = np.ones((TP, TP))
    Sol_next = np.ones((TP, TP))
#Now applying boundary conditions to the nth, nth+1/2, and nth plus 1 time step.
#I feel like this should be done outside the loop, but since I need to do it for the half values I will 
#do it for the others as well.
    mod1 = BC_Modifier(t, phi)
    mod2 = BC_Modifier(t + dt/2, phi)
    mod3 = BC_Modifier(t + dt, phi)
    #Left, is the neumann
    #Right
    Sol[:, -1] = [x*mod1 for x in right_um]
    HVM[:, -1] = [x*mod2 for x in right_um]
    Sol_next[:, -1] = [x*mod3 for x in right_um]
    ###Bottom
    Sol[0, :] = [x*mod1 for x in bottom_um]
    HVM[0, :] = [x*mod2 for x in bottom_um]
    Sol_next[0, :] = [x*mod3 for x in bottom_um]
    ###Top
    Sol[-1, :] = [x*mod1 for x in top_um]
    HVM[-1, :] = [x*mod2 for x in top_um]
    Sol_next[-1, :] = [x*mod3 for x in top_um]
    
#this is the ghost node term that appears in some equations. It is calculate here 
#so it doesnt have to be repeatedly calculated. For my problem its just zero.    
    GNT = b*2*dx*v
    # =============================================================================
    # #Step "A":The t=n to t=n+ 1/2 step
    # =============================================================================
    rhs_a = np.ones(N + 1)
    #rhs for step a is (N+1) because of the ghost node
    #Pre thomas algorithm set up
    av = [a]*(N + 1)
    bv = [b]*(N + 1)
    cv = [c]*(N + 1)
    #different first input because of ghost node
    cv[0] = b + c 
    
    for k in range(1, N + 1):
    ##first eq different    
        rhs_a[0] = -b*Sol[k - 1, 0] + d*Sol[k, 0] - c*Sol[k + 1, 0] + GNT + q[k, 0]
    #    b*v*2*dx from ghost node, even if it is zero
        ##middle eqs
        for j in range(1, N):
            rhs_a[j] = -b*Sol[k - 1, j] + d*Sol[k, j] - c*Sol[k + 1, j] +q[k, j]
    ##last eq different
        rhs_a[-1] = -b*Sol[k - 1,-2] + d*Sol[k,-2] - c*Sol[k + 1,-2] - c*HVM[k,-1] + q[k, -2] 
        HVM[k, 0:-1] = thomas_alg_func(av, bv, cv, rhs_a)
    # =============================================================================
    # #Step "B" the t=n+1/2 to t=n+1
    # =============================================================================
    #must solve one column at a time to preserve tridiagonal structure
    #I would have to modify every Nth element of the b and c vectors, is fine for now
    #okay rhs b still is N long, I just need to do the steps N+1 one times    
    rhs_b = np.ones(N)
    av_b = [a]*(N)
    bv_b = [b]*(N)
    cv_b = [c]*(N)    
    #first all the u(0,k) for k=1,2...N)      
    #these are a little different because of the ghost node.
    #first equation different
    rhs_b[0] = -b*HVM[1, 1]- GNT + d*HVM[1, 0] - c*HVM[1, 1] -b*Sol_next[0, 0]  + q[1, 0]
    #middle eqs
    for k in range(2, N):
        rhs_b[k-1] = -b*HVM[k, 1] - GNT + d*HVM[k, 0] - c*HVM[k, 1] + q[k, 0]
    #last equation different
        rhs_b[-1] = -b*HVM[N, 1] - GNT + d*HVM[N, 0] - c*HVM[N, 1] - c*Sol_next[N + 1, 0] + q[N, 0]
    Sol_next[1:-1, 0] = thomas_alg_func(av_b, bv_b, cv_b, rhs_b)
    
    #now for the next ones, u(j,k) for for j=1,2...N and k=1,2...N)
    for j in range(1, N + 1):
    #first equation different
        rhs_b[0] = -b*HVM[1, j - 1] + d*HVM[1, j] - c*HVM[1, j + 1] -b*Sol_next[0, j] + q[1, j]
    #middle eqs
        for k in range(2, N):
            rhs_b[k-1] = -b*HVM[k, j-1] + d*HVM[k, j] - c*HVM[k, j + 1] + q[k, j]
    #last equation different
        rhs_b[-1] = -b*HVM[N, j - 1] + d*HVM[N, j] - c*HVM[N, j + 1] - c*Sol_next[N + 1, j] + q[N, j]
        Sol_next[1:-1, j] = thomas_alg_func(av_b, bv_b, cv_b, rhs_b)
    return (Sol_next)

def Manufactured_Solution(x, y, t, phi, a_x):
    """This function returns part of the exact solution for the method of manufactured solutions..."""
#I am framing this manufactured solution to be similar to my boundary conditions, for example it contains the same
# term (1-exp(-phi8t)), and also d/dx at x=ax is also zero
#the variable im using is big Q,
#Q(x,y,t)=(1-exp(-phi*t)*(cos(x)*sin(y)+5/4pi(x+0.1)(y+0.1))
#inputs are the arrays x, y, a single t value and the constant phi and the constant ax  
#I modified this so some of the BCS arent always zero. Always zero is no fun
    TP = len(x)
    func_vals = np.zeros((TP, TP))
    for k in range (TP):
        row = [(1 - exp(-phi*t))*(cos(x-a_x)*sin(y[k])+1) for x in x]
        func_vals[k] = row
    return func_vals

#manufactured solution partial Term
def MSPT(x, y, t, phi, a_x):
    """This function returns partial derivative terms for the the method of manufactured solutions..."""
#I am framing this manufactured solution to be similar to my boundary conditions, for example it contains the same
# term (1-exp(-phi8t)), and also d/dx at x=ax is also zero
#the variable im using is big Q,
#Q(x,y,t)=(1-exp(-phi*t)*(cos(x-ax)*sin(y)+5/4*pi*x+9/64*pi*y) 
#inputs are the arrays x, y, a single t value and the constant phi    
    TP = len(x)
    func_vals = np.zeros((TP, TP))
    partial_t = np.zeros(TP)
    partial_x2 = np.zeros(TP)
    fee_tee = -phi*t
    uno = (1 - exp(fee_tee))

    for k in range (TP):
        partial_t[:] = [phi*exp(fee_tee)*(cos(x-a_x)*sin(y[k])+1) for x in x]
        partial_x2[:] = [-uno*cos(x-a_x)*sin(y[k]) for x in x]
#        partial y2 is just the same as partial x2
        row = partial_t - partial_x2 - partial_x2
        func_vals[k] = row
    return func_vals

# =============================================================================
# Main program here 
# =============================================================================
    
# =============================================================================
# Grid Convergence. Carreid out for the first time step.
# =============================================================================
#CLOSE_ENOUGH_GCS = 2.5*10**-2
CLOSE_ENOUGH_GCS = 0.9
max_diff = 2 
#we will start with 8 internal points
N = 8
dt = 0.25
#t=dt
t=  0
t_bc = round(4.605/phi)
steps_bc = round(t_bc/dt) 

while max_diff>CLOSE_ENOUGH_GCS:
#for N
    Sol_N, x, dx, right_um, bottom_um, top_um, mu, a, b, c, d, TP = Set_up(N, dt)
#    this varibale q is for a prescirbed function anad will later overritten for the method of manual solutions.
#    it is out side the set up function for now.
#    the input would be (dt*1/2(q(x,y,t_n)+q(x,y,t_n+1))
    q=np.zeros((TP,TP)) 
#    q=np.ones((TP,TP)) 
    Sol_N_next = ADI(Sol_N, x, dx, right_um, bottom_um, top_um, mu, a, b, c, d, TP, q) 
#for 2N    
    N = N + N
    Sol_2N, x_ex, dx_ex, right_um_ex, bottom_um_ex, top_um_ex, mu_ex, a_ex, b_ex, c_ex, d_ex, TP_ex = Set_up(N, dt)
    q=np.zeros((TP_ex,TP_ex))
#    q=np.ones((TP_ex,TP_ex)) 
    Sol_2N_next = ADI(Sol_2N, x_ex, dx_ex, right_um_ex, bottom_um_ex, top_um_ex, mu_ex, a_ex, b_ex, c_ex, d_ex, TP_ex, q)
# Interpolate the values from the 2N grid so the points match up to the same physical location
#Interpolate the finer grid values to the coarser grid
    the_interloper = interpolate.interp2d(x_ex, x_ex, Sol_2N_next, kind='cubic')
    Sol_int = the_interloper(x, x)
#find the biggest difference between the solutions    
    Diff_Matrix=abs(Sol_N_next[1:-1, 1:-1] - Sol_int[1:-1, 1:-1])
    max_diff=Diff_Matrix.max()
    print('done')
print('all done gcs')    
#fig5, ax5 = plt.subplots()
#plt.grid(1)    
#plt.plot(x,Sol_N_next[1,:],'-b') 
#plt.plot(x_ex,Sol_2N_next[1,:],':r')
#plt.plot(x,Sol_int[1,:],':g')
#Doubling the number of grid points resulted in less than "CLOSE ENOUGH' maximum difference between the two solutions
#We will call this convereged and go back to the previous N value for the steady state solution
#The 2N grid will serve as our "Exact'" Solution
#So that the steady state solution will have N grid points, and the "Exact" solution will have 2N grid points.
N_final=round(N/2)
N_ex=N

# =============================================================================
# Advancing solution forward in time
# =============================================================================
#initial time=0
t = 0
#I actually dont need to call the setup functions again, thet are ready to go/
#setting up the solution and the "exact" solution inputs
#Sol_N, x, dx, right_um, bottom_um, top_um, mu, a, b, c, d, TP = Set_up(N_final, dt)
#subscript_ex for exact
#(Sol_2N, x_ex, dx_ex, right_um_ex, bottom_um_ex, top_um_ex, 
#mu_ex, a_ex, b_ex, c_ex, d_ex, TP_ex) = Set_up(N_ex, dt)

#we will use the L infity  error with the u^tn+1 as the reference values 
Linf_error = 2
#CLOSE_ENOUGH_SS = 2.5*10**-1
CLOSE_ENOUGH_SS = 0.9
##for the finer grid it takes longer to reach that close enough value, lets think about that....
while Linf_error > CLOSE_ENOUGH_SS:
#for m in range(round(steps_bc*6)):
    Sol_N_next = ADI(Sol_N, x, dx, right_um, bottom_um, top_um, mu, a, b, c, d, TP, q)
    Sol_2N_next = ADI(Sol_2N, x_ex, dx_ex, right_um_ex, bottom_um_ex, top_um_ex,
                      mu_ex, a_ex, b_ex, c_ex, d_ex, TP_ex, q)
    G1 = abs(Sol_N_next[1:-1, 1:-1] - Sol_N[1:-1, 1:-1])
    Linf_error = G1.max()
    Sol_N = Sol_N_next    
    Sol_2N = Sol_2N_next  
#advance time
    t = t + dt
    print('HEY THERE')
# =============================================================================
# Calculating error 
# =============================================================================
#   interpolating the finer grid solution to the coarser mesh
the_interloper = interpolate.interp2d(x_ex, x_ex, Sol_2N, kind='cubic')
Sol_int = the_interloper(x, x)    
L1_error = L1_norm(Sol_N[1:-1, 1:-1], Sol_int[1:-1, 1:-1])
L2_error = L2_norm(Sol_N[1:-1, 1:-1], Sol_int[1:-1, 1:-1])

# =============================================================================
# Carry out the Method of Manufactured solution.
# =============================================================================
#The I picked my manufactured solution it sill has a Neumann boundary at x=ax, and teh top left and right
#are called by the same term that the updated problem is,
#So all I need to do is feed in the new un-modified BC's to my ADI function, 
#and also the partial terms which are now the forcing term/ prescribed function 
#getting the unmodified ms solution bondary conditions.  So just feed in a large time value 
Sol_ms, x, dx, right_um, bottom_um, top_um, mu, a, b, c, d, TP = Set_up(32, dt)
right_um_ms = Manufactured_Solution(x, x, 1000, phi, a_x)[:, -1]
#the bottom and top is just zero, but I will keep it around to be general
#... I think I wanna change the function a bit, always zero is no fun
bottom_um_ms = Manufactured_Solution(x, x, 1000, phi, a_x)[0, :]
top_um_ms = Manufactured_Solution(x, x, 1000, phi, a_x)[-1, :]
#okay wait when I changed the function I no longer have nueman boundary conditions let me
#okay let me change again


#now I dont need to call Set Up function, again I will just use the same grid, but I will set up some blank matrices 
#Sol_ms=np.zeros((TP,TP))
#intial condistion is zero for all points, including the boundaries. 
#For this, I already have the teh analytical solution 
t=0
Linf_error_ms = 2
#CLOSE_ENOUGH_SS = 2.5*10**-1
CLOSE_ENOUGH_SS = 0.001
#setting up the forcing fuinction term
while Linf_error_ms > CLOSE_ENOUGH_SS:
    q=(1/2)*dt*(MSPT(x, x, t, phi, a_x)+MSPT(x, x, t+dt, phi, a_x))
    Sol_ms_next = ADI(Sol_ms, x, dx, right_um_ms, bottom_um_ms, top_um_ms, mu, a, b, c, d, TP, q)
    Sol_ms_ex = Manufactured_Solution(x, x, t+dt, phi, a_x)
    G1 = abs(Sol_ms_next[1:-1, 1:-1] - Sol_ms[1:-1, 1:-1])
    Linf_error_ms = G1.max()
    Sol_ms = Sol_ms_next    
#advance time
    t = t + dt
    print('MS') 
#And we get the exact solution back on the boundaries because those are prescribed, elsewhere,
#Like the Nuemann boundary we get the same shape but with dampening






# ==============================================================================
# Plotting
# =============================================================================
#bottom
#y=x
#fig5, ax5 = plt.subplots()
#plt.grid(1)    
#plt.plot(y,Sol_next[0,:],'-b') 
#plt.plot(y,Sol_next[1,:],':r')
#plt.xlabel('x')
#plt.ylabel('u(x, y)')
#ax5.legend(['u(x, a_y)','u(x, y_1)'])
#ax5.title.set_text('BOTTOM At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(t,4),round(dx,4),round(dt,4),round(mu,4)))  
##top
#fig6, ax6 = plt.subplots()
#plt.grid(1)    
#plt.plot(y,Sol_next[-1,:],'-b') 
#plt.plot(y,Sol_next[-2,:],':r')
#plt.xlabel('x')
#plt.ylabel('u(x, y)')
#ax6.legend(['u(x, b_y)','u(x, y_N)'])
#ax6.title.set_text('TOP: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(t,4), round(dx,4),round(dt,4),round(mu,4)))  
##left
#fig7, ax7 = plt.subplots()
#plt.grid(1)
#plt.plot(x,Sol_next[:,0],'-b') 
#plt.plot(x,Sol_next[:,1],':r')
#plt.xlabel('y')
#plt.ylabel('u(x, y)')
#ax7.legend(['u(a_x, y)','u(x_1, y)'])
#ax7.title.set_text('LEFT: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(t,4), round(dx,4),round(dt,4),round(mu,4)))  
##right
#fig8, ax8 = plt.subplots()
#plt.grid(1)    
#plt.plot(x,Sol_next[:,-1],'-b') 
#plt.plot(x,Sol_next[:,-2],':r')     
#plt.xlabel('y')
#plt.ylabel('u(x, y)')
#ax8.legend(['u(b_x, y)','u(x_N, y)'])  
#ax8.title.set_text('RIGHT: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(t,4), round(dx,4),round(dt,4),round(mu,4)))
#
#3D GRAPH
from mpl_toolkits import mplot3d
fig10 = plt.figure()
ax10 = plt.axes(projection='3d')
BIGX, BIGY = np.meshgrid(x, y)    
from matplotlib import cm    
surf = ax10.plot_surface(BIGX, BIGY, Sol_next, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
#fig2.colorbar(surf, shrink=0.75, aspect=5)
ax10.set_title('u(x,y,t=%s s) \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s '%(round(t,4),round(dx,4),round(dt,4),round(mu,4)))
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


# =============================================================================
# GCS Plotting
# =============================================================================
#t=dt
#fig8, ax8 = plt.subplots()
#plt.grid(1)    
#plt.plot(x,Sol_N_next[:,1],'-b') 
#plt.plot(x_ex,Sol_2N_next[:,1],':r')     
#plt.xlabel('y')
#plt.ylabel('u(x, y)')
##ax8.legend(['u(b_x, y), N GridPoints','u(b_x, y) 2N grid points'])  
#ax8.title.set_text('RIGHT: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
#                   %(round(t,4), round(dx,4),round(dt,4),round(mu,4)))

#t=dt
fig8, ax8 = plt.subplots()
plt.grid(1)    
plt.plot(x,Sol_ms[5,:],'-b') 
plt.plot(x,Sol_ms_ex[5,:],':r')     
plt.xlabel('y')
plt.ylabel('u(x, y)')
#ax8.legend(['u(b_x, y), N GridPoints','u(b_x, y) 2N grid points'])  
ax8.title.set_text('Left: At t= %s s \n $\Delta$x=$\Delta$y=%s units, $\Delta$ t= %s s, $\mu$= %s'
                   %(round(t,4), round(dx,4),round(dt,4),round(mu,4)))