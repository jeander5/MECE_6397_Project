# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:19:12 2020

@author: johna
"""

#PTRANSI, Pentadiagonal Linear System of equation solver,
#Source: https://www.hindawi.com/journals/mpe/2015/232456/

#This version just contains the function, and not all my comments from when 
#I was writing the code and figure out the mistake from the website
#The websites equations were correct, but there was a mistake in their psuedocode that I was following


#imports, should this maybe inside the function? I mean yes, but what is proper?
import numpy as np
import numpy.linalg as lin

#The Function
def PTRANSI(P,RHS):
    """solves a pentadiagonal linear system of equation"""
#Inputs are pentadiagonal matrix P and RHS vector
#Px=RHS, returns solution vector x
#e,c sub diagonal, d main diagonal, a, b superdiagonal.     

#getting length N, I will just do this at the beginning
    N=len(RHS)

#getting diagonals
#(e,c,d,a,b,0....0)
    d=P.diagonal()
    a=P.diagonal(1)
    b=P.diagonal(2)
    c=P.diagonal(-1)
    e=P.diagonal(-2)
       
#adding zeros at begginning of sub diagonalas and end of super diagaonals
#Now all diagonal vectors are the same length.    
    e=np.append([0,0],[e])
    c=np.append([0],[c])
    a=np.append([a],[0])
    b=np.append([b],[0,0])   
    
#The algorithm

#Step 1 get determinat
#G is just Generic placeholder variable
#    G=np.linalg.det(P)
#Step 2 if det(P) != 0 proceed.    
#    if G==0:
#        print('Tough luck pal! find a new algorithm!)
#        break      
     
#set up algorithm vectors
    mu=np.zeros(N)
    alpha=np.zeros(N-1)
    beta=np.zeros(N-2)
    gamma=np.zeros(N)
    z=np.zeros(N)
#x is the solution
    x=np.zeros(N)

#im just following the psuedo code here
#this algorithm transforms the original pentadiagonal matrix using row operations

#step 3, zeroth elements
    mu[0]=d[0] 
    alpha[0]=a[0]/mu[0]
    beta[0]=b[0]/mu[0]
    z[0]=RHS[0]/mu[0]

##step 4, first elements
    gamma[1]=c[1]
    mu[1]=d[1]-alpha[0]*gamma[1]
    alpha[1]=(a[1]-beta[0]*gamma[1])/mu[1]
    beta[1]=b[1]/mu[1]
    z[1]=(RHS[1]-z[0]*gamma[1])/mu[1]
##step 5, the internal elements
    for j in range(2,N-2):
        gamma[j]=c[j]-alpha[j-2]*e[j]
        mu[j]=d[j]-beta[j-2]*e[j]-alpha[j-1]*gamma[j]
        alpha[j]=(a[j]-beta[j-1]*gamma[j])/mu[j]
        beta[j]=b[j]/mu[j]
        z[j]=(RHS[j]-z[j-2]*e[j]-z[j-1]*gamma[j])/mu[j]

##after the loop, last two elements  
    gamma[-2]=c[-2]-alpha[-3]*e[-2]
    mu[-2]=d[-2]-beta[-2]*e[-2]-alpha[-2]*gamma[-2]
    alpha[-1]=(a[-2]-beta[-1]*gamma[-2])/mu[-2]
    gamma[-1]=c[-1]-alpha[-2]*e[-1]
    mu[-1]=d[-1]-beta[-1]*e[-1]-alpha[-1]*gamma[-1]
    z[-2]=(RHS[-2]-z[-4]*e[-2]-z[-3]*gamma[-2])/mu[-2]
    z[-1]=(RHS[-1]-z[-3]*e[-1]-z[-2]*gamma[-1])/mu[-1]

##okay so everything above basically just transformed our original P using row operations
##now we can easily solve for x
##step 6, solving x
    x[-1]=z[-1]
    x[-2]=z[-2]-alpha[-1]*x[-1]
    for j in range(1,N-1):
        x[-2-j]=z[-2-j]-alpha[-1-j]*x[-1-j]-beta[-j]*x[-j]  
    return x
