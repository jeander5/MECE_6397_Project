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

#RIGHT: This is a the big one 
#g_a(bx) +(y-a_y)/(b_y-a_y) *[f_a(b_x)-g_a(b_x)]

#that line reminds me of how alot of the non dimensional equations end up
#like (T-To)/(T1-To)

#given functions
#g_a and f_a I will just call g and f
#g=(x-a_x)^2*cos(x)
#f=x*(x-a_x)^2  

#Initial conditions, are just zero for all points, INSIDE the boundary
#U(x,y,t)=0
#I will still define this.
Uo=0

#I want to really make things as general as possible, so I could easily adapt this do a different problem.
#that may mean making functions for things that dont  necessarily need functions
#and reassigning zero as zero.
#I want this easily adaptable

#Functions

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
#it will just have to call the other functions
# im not liking this tho, it will have to call those other functions with just a single 
# y value
# and i like my functions to be self contained.....
#it doesnt have to call those other functions...... 
#because its just a constant
#I will just define those inside the function.

def RIGHT(y, a_x, a_y, b_x, b_y):
    """returns the RIGHT Boundary conditions values for this problem"""    
#inputs are the boundary points of the domainand the discretized y values
# I will just break this equation up for now     
    uno = cos(b_x)*(b_x-a_x)*(b_x-a_x)
    dos = [(y-a_y)/(b_y-a_y) for y in y]
    tres = b_x*(b_x-a_x)*(b_x-a_x)-uno
    func_vals= [uno + dos*tres for dos in dos]
    return func_vals  

#ok next

#Number of discretized points

#Number of internal x points
N_x=8    
#Number of internal y points
N_y=8    
    
#calling the DIF
x, dx = DIF(b_x,N_x)
y, dy = DIF (b_y,N_y)

#lengths, defined here so they wont need to be calculated else where
len_x=len(x)
len_y=len(y)

#setting up solution matrix....I really need a 3D matrix
#Do i though...yes If I want to be able to plot the x and y for any time t
#but if i just want the steady steady state I geuss I really only need 2 2D matrices
#one for time n and one for time n+dt
#hmmmmmm amd I dont know the the length of the 3rd dimension either.....

#this kinda seems like an important point. I can solve it with 2 matrices.
# but how will I be able to plot every u(x,y,t), I mean, I can make a matrix that keeps
# on extending in the third dimension but that is inefeccient
# I geuss I can just solve it, determine the length of third dimension. then solve it 
#again and store it. but that seems like Im doing twice the calculations.

#And im gonna need a while loop too.

#hmm lets think about this......I wont really ever need to plot all the values,
#only select values which I could just call again as needed....
#No but If say i wanted to plot the middle of the time interval I would need to start all the 
#way over to reach that value
#the self appening matrix seems like a good idea, even if it is inefficient.
#or is solving and then resolving better? I dont know
#what is worse, running all the algorithms an additional time or 
#continuolsy reallocating storage space for the 3rd dimension of the matrix
#probalby the second option is worse.

#okay that is how Im gonna proceed.
#Sol_1 is my solution matrix for the nth time step and Sol 2 for the nth+1 time step
#if Sol_1 appx Sol_@, steady state  is reached.
#if not Sol_2 becomes Sol_1, and new Sol_2 is calulated

#setting up solution matrices
#Sol is my solution matrix for the nth time step and 
#note im just using ones now so that when I apply the ICS I can see that it worked
Sol = np.ones((len_x,len_y))
##Sol next for the nth+1 time step
#Sol_next = np.zeros((len_x,len_y))
#why do I even use that shape function? idk it seems unnessary really.
#Sol=np.zeros(shape=(len_t, len_x))

#okay now I can actually start applying the numerical scheme

#Applying Boundary Conditions
#maybe I should make my solution matrices N_x by N_y so i dont have to continuosly
#re apply the BCs.
#And I see a better way to do this already, only make a new matrix when im applying the scheme.

#And Im gonna have that same issue when I was making that maze script
#the [0,0] element applies to the bottom left corner of the domain, 
#but appears in the upper left corner of the matrix....
#I would kinda like to work backwards but that I feel will be confusing to others
#I can always just flip the matrix later.

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
#Are already defined but I will just write it out and comment it
# I will just redefine over this
Sol[1:-1,1:-1]=Uo
#now I have a row of ones that I will need to change suing the ghost node method

##now I will define Sol_next
##Sol next for the nth+1 time step
##Sol_next = Sol Nope this wont work its the muteablilty thing in python
##if i do this it treats them as the same object
##thats one part of python I dont appreciate
##Ok I found a different way copy()
#
Sol_next=Sol.copy()

#Okay on to the scheme! Do I wanna do CN in 2D with time, which would be a pentadiagonal
#or the ADI, which would be essentially doing a tridiagonal twice.

#I kinda wanna do pentadiagonal because I havent done that before.
#im tired of the thomas algorithm

#I could do both really.

#the ADI is "elegant".

#Im seeing this PTRANS-1 pop up, by some Saudis.
#also a TW, two way approach "d designed for pairs of parallel computers"
#Im gonna try this PTRANS-1 it looks pretty good















# =============================================================================
# different versions of given_x   
# =============================================================================
# =============================================================================
# Dont worry abou this type of stuff for now. I do wanna try to implement these type of things later tho.   
# =============================================================================
##version 2 does this take advantage of parrelism?
##    I dont think so so actually, because its not in a for loop, its in the "inline python" for loop
##    uno=[x-a for x in x]
##    dos=[uno*uno for x in uno]
##    tres=[x*dos for x in x]     
## version 3, doing the expansion my self, this i dont actually think is better
##    func_vals=[x*x*x-2*a*x*2+a*a*x for x in x]
##version 4 using a real for loop
##    lenx=len(x)
## for j in range (lenx)
#    xi=x[j]
#    uno=(xi-a)
#    dos=uno*uno
#    tres=xi*dos
##and then I could also do loop unrolling
##this seems like it is correct, it doesnt have to search the x vector multiple times per loop
##it should take advantage of parrelism.
#and remember the compiler does alot of this type of stuff automatically for me.     



    
     
