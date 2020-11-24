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
y[:]=np.array([8,33,8,24,29,98,99,17,57,108])

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
mu=np.zeros(10)
alpha=[0]*(leny-1)
beta=[0]*(leny-2)
gamma=[0]*leny
z=[0]*leny
newz=[0]*leny
#x is the solution
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
    
    gamma[-2]=c[-2]-alpha[-3]*e[-2]
    mu[-2]=d[-2]-beta[-2]*e[-2]-alpha[-2]*gamma[-2]
    alpha[-1]=(a[-2]-beta[-1]*gamma[-2])/mu[-2]
    gamma[-1]=c[-1]-alpha[-2]*e[-1]
    mu[-1]=d[-1]-beta[-1]*e[-1]-alpha[-1]*gamma[-1]
    z[-2]=(y[-2]-z[-4]*e[-2]-z[-3]*gamma[-2])/mu[-2]
    z[-1]=(y[-1]-z[-3]*e[-1]-z[-2]*gamma[-1])/mu[-1]

##note im using [leny-1] everywhere rather than [-1] bc the greek vectors arent all the same length
##And because im being kinda sloppy with my list and arrays z is an array of 1x1 arrays
##lol, also x is now to but I will fix this
# Still need to fix this
    
#
##okay so everything above basically just transformed our original P using row operations
##now we can easily solve P
#
##step 6 solving
x[-1]=z[-1]
x[-2]=z[-2]-alpha[-1]*x[-1]
for j in range(1,leny-1):
    x[-2-j]=z[-2-j]-alpha[-1-j]*x[-2-j+1]-beta[-j]*x[-2-j+2]  
print(x)
ANSWER=lin.solve(P,y)
print('\n')
print(ANSWER)

#ok I now need to make a function of this.
def PTRANSI(P,RHS):
    """solves a pentadiagonal linear system of equation"""
#Inputs are pentadiagonal matrix P and RHS vector
#Px=RHS, returns solution vector x
#e,c sub diagonal, d main diagonal, a, b superdiagonal.     

#getting length, I will just do this at the beginning
    leny=len(RHS)# i still have this as leny CHANGE IT LATER    


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
    G=np.linalg.det(P)
#Step 2 if det(P) != 0 proceed.    
#    if G==0:
#        print('Tough luck pal! find a new algorithm!)
#        break      
    
#well if Im gonna check the determinant I need to remake the matrix or just have the input be the matrix.              
#probably should just input the matrix.
#I will need to be able to fill up a penta diagonal matrix efficiently
#and im not seeing a nice numpy function for filling sub and super diagonals    
# something like this works:
#a = np.zeros((5, 5))
#b = np.ones(3)
#c= np.ones(2)
#np.fill_diagonal(a[1:], 123*b)#sub sub
#np.fill_diagonal(a[:,1:], -321*b)#sub
#np.fill_diagonal(a,444)#main
#np.fill_diagonal(a[2:], 99*c)#super
#np.fill_diagonal(a[:,2:], -77*c)#super super    

#why dont I wanna make the full diagonal matrix?
#The input should be the full matrix to keep it generic.    
    
#set up algorithm vectors
    mu=np.zeros(leny)
    alpha=np.zeros(leny-1)
    beta=np.zeros(leny-2)
    gamma=np.zeros(leny)
    z=np.zeros(leny)
#x is the solution
    x=np.zeros(leny)

#im just following the psuedo code here
#this algorith transforms the original penta diagonal matrix using row operations

#step 3, first elements
    mu[0]=d[0] 
    alpha[0]=a[0]/mu[0]
    beta[0]=b[0]/mu[0]
    z[0]=y[0]/mu[0]

##step 4, second elements
    gamma[1]=c[1]
    mu[1]=d[1]-alpha[0]*gamma[1]
    alpha[1]=(a[1]-beta[0]*gamma[1])/mu[1]
    beta[1]=b[1]/mu[1]
    z[1]=(y[1]-z[0]*gamma[1])/mu[1]
#
##step 5, for i=3,4,...n-2, the internal elements
##so for 2-n-3
    for j in range(2,leny-2):
        gamma[j]=c[j]-alpha[j-2]*e[j]
        mu[j]=d[j]-beta[j-2]*e[j]-alpha[j-1]*gamma[j]
        alpha[j]=(a[j]-beta[j-1]*gamma[j])/mu[j]
        beta[j]=b[j]/mu[j]
        z[j]=(y[j]-z[j-2]*e[j]-z[j-1]*gamma[j])/mu[j]

##after the loop, last two elements  
    gamma[-2]=c[-2]-alpha[-3]*e[-2]
    mu[-2]=d[-2]-beta[-2]*e[-2]-alpha[-2]*gamma[-2]
    alpha[-1]=(a[leny-2]-beta[-1]*gamma[-2])/mu[-2]
    gamma[-1]=c[-1]-alpha[-2]*e[-1]
    mu[-1]=d[-1]-beta[-1]*e[-1]-alpha[-1]*gamma[-1]
    z[-2]=(y[-2]-z[-4]*e[-2]-z[-3]*gamma[-2])/mu[-2]
    z[-1]=(y[-1]-z[-3]*e[-1]-z[-2]*gamma[-1])/mu[-1]

##note im using [leny-1] everywhere rather than [-1] bc the greek vectors arent all the same length
##And because im being kinda sloppy with my list and arrays z is an array of 1x1 arrays
##lol, also x is now to but I will fix this
# Still need to fix this
    
#
##okay so everything above basically just transformed our original P using row operations
##now we can easily solve P
#
##step 6 solving
    x[-1]=z[-1]
    x[-2]=z[-2]-alpha[-1]*x[-1]
    for j in range(1,leny-1):
        x[-2-j]=z[-2-j]-alpha[-1-j]*x[-1-j]-beta[-j]*x[-j]  
    return x
t=PTRANSI(P,y)   
#okay the function works.....
#I just wanna fix the list/array issue. I dont want a list of arrays.
#    I just need to change the z list I think,
#    I dont want that double array thing either, the double brackets.
#okay instead of going greek=[0]*leny just make a numpy array
#Fixed    
#okay I also wann change things like z[-2] to z[-2], the issue is alpha and beta arent the same length 
#as my other vectors    
#alpha[leny-2]=alpha[-1]
#alpha is one element shorter
#beta[leny-3]=beta[-1]
#beta is two elements shorter  
