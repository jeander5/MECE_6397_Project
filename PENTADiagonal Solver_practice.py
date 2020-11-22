# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:50:26 2020

@author: johna
"""


#penta diagonal practice solver
#PTRANS-I
import numpy as np
import numpy.linalg as lin
P=np.zeros((10,10))
P[0,0:3]=[1,2,1]
P[1,0:4]=[3,2,2,5]
P[2,0:5]=[1,2,3,1,-2]
P[3,1:6]=[3,1,-4,5,1]
P[4,2:7]=[1,2,5,-7,5]
P[5,3:8]=[5,1,6,3,2]
P[6,4:9]=[2,2,7,-1,4]
P[7,5:]=[2,1,-1,4,-3]
P[8,6:]=[2, -2,1, 5]
P[9,7:]=[-1,4,8]
#Ok this is correct
y=np.zeros((1,10))
y[:]=[8,33,8,24,29,98,99,17,57,108]

y=np.transpose(y)
leny=len(y)
#getting diagonals
#(e,c,d,a,b,0....0)
d=P.diagonal()
a=P.diagonal(1)
b=P.diagonal(2)
c=P.diagonal(-1)
e=P.diagonal(-2)

#need to modify these slighty, add zeros at begginning of sub diagonalas and 
#end of super diagaonals
a=np.append([a],[0])
b=np.append([b],[0,0])
c=np.append([0],[c])
e=np.append([0,0],[e])

#the algorithm

#Step 1 get determinat
#G is just Generic placeholder variable
G=np.linalg.det(P)
#Step 2 if det(P) != 0 proceed.

#ALGORITHM GREEK VARIABLES
#mu, alpha, beta, gamma
#y is righthand side, z is defined by y

#set up algorithm vectors
mu=[0]*leny
alpha=[0]*(leny-1)
beta=[0]*(leny-2)
gamma=[0]*leny
z=[0]*leny
newz=[0]*leny
#xc is the solution
x=[0]*leny

#im just following the psuedo code here
#step 3,
mu[0]=d[0] 
alpha[0]=a[0]/mu[0]
beta[0]=b[0]/mu[0]
z[0]=y[0]/mu[0]

##step 4
gamma[1]=c[1]
mu[1]=d[1]-alpha[0]*gamma[1]
alpha[1]=(a[1]-beta[0]*gamma[1])/mu[1]
beta[1]=b[1]/mu[1]
z[1]=(y[1]-z[0]*gamma[1])/mu[1]
#
##step 5, for i=3,4,...n-2
##so for 2-n-3
for j in range(2,leny-2):
    gamma[j]=c[j]-alpha[j-2]*e[j]
    mu[j]=d[j]-beta[j-2]*e[j]-alpha[j-1]*gamma[j]
    alpha[j]=(a[j]-beta[j-1]*gamma[j])/mu[j]
    beta[j]=b[j]/mu[j]
    z[j]=(y[j]-z[j-2]*e[j]-z[j-1]*gamma[j])/mu[j]
##after the loop
    
gamma[leny-2]=c[leny-2]-alpha[leny-4]*e[leny-2]
mu[leny-2]=d[leny-2]-beta[leny-4]*e[leny-2]-alpha[leny-3]*gamma[leny-2]
alpha[leny-2]=(a[leny-2]-beta[leny-3]*gamma[leny-2])/mu[leny-2]
gamma[leny-1]=c[leny-1]-alpha[leny-3]*e[leny-1]
mu[leny-1]=d[leny-1]-beta[leny-3]*e[leny-1]-alpha[leny-2]*gamma[leny-1]
z[leny-2]=(y[leny-2]-z[leny-4]*e[leny-2]-z[leny-3]*gamma[leny-2])/mu[leny-2]
z[leny-1]=(y[leny-1]-z[leny-3]*e[leny-1]-z[leny-2]*gamma[leny-1])/mu[leny-1]

##note im using [leny-1] everywhere rather than [-1] bc the greek vectors arent all the same length
##And because im being kinda sloppy with my list and arrays z is an array of 1x1 arrays
##lol, also x and y is now to but I will fix this
# Still need to fix this
    
#
##okay so everything above basically just transferemed our original P using row operations
##now we can easily solve P
#
##step 6 solving
x[leny-1]=z[leny-1]
x[leny-2]=z[leny-2]-alpha[leny-2]*x[leny-1]
for j in range(1,leny-1):
    x[leny-2-j]=z[leny-2-j]-alpha[leny-2-j]*x[leny-2-j+1]-beta[leny-2-j]*x[leny-2-j+2]
print(x)
ANSWER=lin.solve(P,y)
print('\n')
print(ANSWER)

#ok I now need to make a function of this.