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
#y[:]=[8,33,8,24,29,82,71,17,57,108]
y[:]=[8,33,8,24,29,98,99,17,57,108]
#i think they just used the wrong picture or whatever.
#on the website. probably the actual pier review paper has the correct figure I would hope

y=np.transpose(y)
leny=len(y)
#getting diagonals
d=P.diagonal()
a=P.diagonal(1)
b=P.diagonal(2)
c=P.diagonal(-1)
e=P.diagonal(-2)
#NOTE these are not all the same length
#okay they need too all be the same length
#need to modify these slighty, add zeros at begginning of sub diagonalas and 
#end of super diagaonals
a=np.append([a],[0])
b=np.append([b],[0,0])
c=np.append([0],[c])
e=np.append([0,0],[e])

#I dont like the way they order this: e,c,d,a,b
#oh well


#the algorithm

#Step 1 get determinat
#G is just Generic placeholder variable
G=np.linalg.det(P)
#Step 2 if det(P) != 0 proceed.

#ALGORITHM GREEK VARIABLES
#mu, alpha, beta, gamma
#then they also have y and z which im not sure where they come from
#Y is righthand side, z is defined by y

#Do all my greek variable need to be vector of length N
#I think they all do, no wait not all length N

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
z[leny-2]=(y[leny-2]-z[leny-3]*e[leny-2]-z[leny-3]*gamma[leny-2])/mu[leny-2]
z[leny-1]=(y[leny-1]-z[leny-2]*e[leny-1]-z[leny-2]*gamma[leny-1])/mu[leny-1]
#
# =============================================================================
# #putting the new zs seperate in there own loop, following the equations and not the psuedo code
# =============================================================================
newz[0]=y[0]/mu[0]
newz[1]=(y[1]-newz[0]*gamma[1])/mu[1]
for j in range(2,leny):
    newz[j]=(y[j]-newz[j-2]*e[j]-newz[j-1]*gamma[j])/mu[j]
#okay now newz doesnt equal oldz, newz is correct.
#so something was wrong with there psuedo code, or the way I wrote it......
#    
#z=newz    

##note im using [leny-1] everywhere rather than [-1] bc the greek vectors arent all the same length
##And because im being kinda sloppy with my list and arrays z is an array of 1x1 arrays
##lol, also x and y is now to but I will fix this
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
##ok so I have a mistake somewhere 
ANSWER=lin.solve(P,y)
print('\n')
print(ANSWER)
#
##okay i inputed one of the y elements wrong, I wrote 82 for somereason should have been no wait a second
##i inputted it right its just they write the wrong value later...they change 82 and 71 to 98 and 99???
#did they use the wrong picture?????
#
##okay i still ahve an error, the answer should be 1,2,3,4, etc
##ok may last two mu elements are wrong......
##ok i had d[leny-1] when it should have been d[leny-2]
#
##ok now all my mus are correct but I still have an error somewhere
#
##hmmm everything looks right. and my mus are all correct.
#
##if my mus are correct then....my
##d,
##beta
##e
##alpha
##gamma are correct
##okay not necessarily, because I see my gamma is actually incorrect
#they are little sloppy in some parts how they define gamma, how they define beta , ect, ect
#I think that is throwing something off....
#but my mus are all correct
#they define beta as Beta=(beta1,beta2, ...beta_n-2)
#bu then they also say beta_i=b_i/mu_i, for i=2,3,n-1
#similarly for gamma, is just a little confusing to have the first element in the vector be defined as gamma 2.
#so does the i refer to the positon in the vector or the subscript on the symbol?
#im always assuming the subscript
#but then if u do that beta 1 is never defined based on their (4)
#this isnt the issue but is just a little confusing.

#the issue is somewhere with my z's inside that for loop
#step 5 a
#no that isnt it either
#if I set x[-1] and x[-2] to 10 and 9 I get the right answer
#so that loop is correct. 

#Its the z's after the forloop that are wrong
#but they look correct.
#maybe THEY have a sign error somewhere

#i really think its there psuedo code...or I made a mistake somewhere.... 
#I will now make newz following their (5)
#Boom! okay so what is the difference between these lines....
#old Z, following the psuedo code
#z[leny-2]=(y[leny-2]-z[leny-3]*e[leny-2]-z[leny-3]*gamma[leny-2])/mu[leny-2]
#z[leny-1]=(y[leny-1]-z[leny-2]*e[leny-1]-z[leny-2]*gamma[leny-1])/mu[leny-1]
#new z, following euation (5)
#for j in range(leny-2,leny):
#    newz[j]=(y[j]-newz[j-2]*e[j]-newz[j-1]*gamma[j])/mu[j]
#its totally their psuedo code!!!!!!!!
#they use z[n-1]=(z[n-2]*e[n-1]-z[n-2]*gamma[n-1])/mu[n-1]
#when it needs to need to be.....
#z[n-1]=(z[n-3]*e[n-1]-z[n-2]*gamma[n-1])/mu[n-1]    
#bastards! I should email them.