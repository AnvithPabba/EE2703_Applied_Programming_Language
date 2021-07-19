import numpy as np
import scipy
from scipy.integrate import *
import matplotlib
from matplotlib.pyplot import * 
import math

#defining fuction for exponential
def f(x):
	return np.exp(x)

#defining function for cos(cos(x))
def g(x):
	return np.cos(np.cos(x))

def f_periodic(x):
	return f(x%(2*(np.pi)))

def g_periodic(x):
	return g(x%(2*(np.pi)))

#test_list = [2,2]
#test_array = np.array(test_list)
#print(f(test_array))

#x-axis vector ranges from -2*pi to 4*pi with increments of 0.1
x = np.arange(-2*(np.pi),4*(np.pi),0.1)
y1 = f(x)
y2 = g(x)
y3 = f_periodic(x)
y4 = g_periodic(x)

#plotting 1st graph
xticks([-2*(np.pi),-1*(np.pi),0*(np.pi),1*(np.pi),2*(np.pi),3*(np.pi),4*(np.pi)])
xlabel('$x\longrightarrow$')
ylabel('$exp(x),log scale\longrightarrow$')
title('Plot of original and expected fourierseries exponent with logy scale')
yscale('log')
plot(x , y1, color = 'green', label = 'original exp(x)')
plot(x , y3, color = 'blue', label = 'exp(x) periodic extension')
legend(loc = 'upper right')
grid()
show()

#plotting 2nd graph
xticks([-2*(np.pi),-1*(np.pi),0*(np.pi),1*(np.pi),2*(np.pi),3*(np.pi),4*(np.pi)])
xlabel('$x\longrightarrow$')
ylabel('$cos(cos(x))\longrightarrow$')
title('Plot of original and fourierseries expected coscos')
plot(x , y2, color = 'blue', label = 'original cos(cos(x)')
plot(x , y4, color = 'red', label = 'cos(cos(x)) periodic extension')
legend(loc = 'upper right')
grid()
show()

#defining u1(x,k) and v1(x,k)
def u1(x,k):
	return f(x)*(np.cos(k*x))  #for a_n of exp(x)

def v1(x,k):
	return f(x)*(np.sin(k*x))  #for b_n of exp(x)


A = np.empty([1,26], dtype = object)
B = np.empty([1,26], dtype = object)


#finding first 51 coeff of the above
for k in range(1,26):
	A[0][k] = (1/(np.pi))*quad(u1,0,2*(np.pi),args=(k))[0]
	B[0][k] = (1/(np.pi))*quad(v1,0,2*(np.pi),args=(k))[0]

A[0][0] = (1/(2*(np.pi)))*quad(f,0,2*(np.pi))[0]
B[0][0] = (1/(2*(np.pi)))*quad(f,0,2*(np.pi))[0]

C1 = np.empty([1,51], dtype = object)
C1[0][0] = A[0][0]

for i in range(1,26):
	C1[0][2*i -1] = A[0][i]
for j in range(1,26):
	C1[0][2*j] = B[0][j]


x = np.arange(0,51)

title('Semilog plot of fourier coeff of exp(x)')
yscale('log')
scatter(x , np.absolute(C1[0]), color='red', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

title('loglog plot of fourier coeff of exp(x)')
yscale('log')
xscale('log')
xlim([0.9,80])
scatter(x , np.absolute(C1[0]), color='red', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

#Same thing as above now for cos cos

#defining u2(x,k) and v2(x,k)
def u2(x,k):
	return g(x)*(np.cos(k*x))  #for a_n of coscos

def v2(x,k):
	return g(x)*(np.sin(k*x))  #for b_n of coscos

A = np.empty([1,26], dtype = object)
B = np.empty([1,26], dtype = object)


#finding first 51 coeff of the above
for k in range(1,26):
	A[0][k] = (1/(np.pi))*quad(u2,0,2*(np.pi),args=(k))[0]
	B[0][k] = (1/(np.pi))*quad(v2,0,2*(np.pi),args=(k))[0]

A[0][0] = (1/(2*(np.pi)))*quad(g,0,2*(np.pi))[0]
B[0][0] = (1/(2*(np.pi)))*quad(g,0,2*(np.pi))[0]

C2 = np.empty([1,51], dtype = object)
C2[0][0] = A[0][0]

for i in range(1,26):
	C2[0][2*i -1] = A[0][i]
for j in range(1,26):
	C2[0][2*j] = B[0][j]


x = np.arange(0,51)
#print(x)
#print(C[0])

title('Semilog plot of fourier coeff of cos(cos(x))')
yscale('log')
scatter(x , np.absolute(C2[0]), color='red', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

title('loglog plot of fourier coeff of cos(cos(x))')
yscale('log')
xscale('log')
xlim([0.9,80])
scatter(x , np.absolute(C2[0]), color='red', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

#Now for the "Least square approach"

x=np.linspace(0,2*(np.pi),401)
x=x[:-1] # drop last term to have a proper periodic integral
b1=f(x) # f has been written to take a vector
b2=g(x)
A=np.zeros((400,51)) # allocate space for A
A[:,0]=1 # col 1 is all ones
for k in range(1,26):
	A[:,2*k-1]=np.cos(k*x) # cos(kx) column
	A[:,2*k]=np.sin(k*x) # sin(kx) column
#endfor
c1=np.linalg.lstsq(A,b1,rcond = None)[0] # the ’[0]’ is to pull out the
c2=np.linalg.lstsq(A,b2,rcond = None)[0] # best fit vector. lstsq returns a list.

#now plotting the coeff we got through lstsq approach
#for exp
x = np.arange(0,51)

title('Semilog plot of lstsq fourier coeff of exp(x)')
yscale('log')
scatter(x , np.absolute(c1), color='green', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

title('loglog plot of lstsq fourier coeff of exp(x)')
yscale('log')
xscale('log')
xlim([0.9,80])
scatter(x , np.absolute(c1), color='green', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

#for coscos
title('Semilog plot of lstsq fourier coeff of cos(cos(x))')
yscale('log')
scatter(x , np.absolute(c2), color='green', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

title('loglog plot of lstsq fourier coeff of cos(cos(x))')
yscale('log')
xscale('log')
xlim([0.9,80])
scatter(x , np.absolute(c2), color='green', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

#Comparision plots

x = np.arange(0,51)

title('Semilog plot, comparision of fourier coeff of exp(x)')
yscale('log')
scatter(x , np.absolute(c1), color='green', marker='o', label = 'exp(x)_coeff')
scatter(x , np.absolute(C1), color='red', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

title('loglog plot, comparision of fourier coeff of exp(x)')
xlim([0.9,80])
yscale('log')
xscale('log')
scatter(x , np.absolute(c1), color='green', marker='o', label = 'exp(x)_coeff')
scatter(x , np.absolute(C1), color='red', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

title('Semilog plot, comparision of fourier coeff of cos(cos(x))')
yscale('log')
scatter(x , np.absolute(c2), color='green', marker='o', label = 'exp(x)_coeff')
scatter(x , np.absolute(C2), color='red', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()

title('loglog plot, comparision of fourier coeff of cos(cos(x))')
yscale('log')
xscale('log')
xlim([0.9,80])
scatter(x , np.absolute(c2), color='green', marker='o', label = 'exp(x)_coeff')
scatter(x , np.absolute(C2), color='red', marker='o', label = 'exp(x)_coeff')
legend(loc = 'upper right')
grid()
show()


#to find maximum absolute deviation

c1_dev = np.absolute(c1 - C1)
c2_dev = np.absolute(c2 - C2)

def list_max(A):
	max = 0
	for i in range(len(A[0])):
		if A[0][i] > max:
			max = A[0][i]
	return max

c1_dev_max = list_max(c1_dev)
print(c1_dev_max)
c2_dev_max = list_max(c2_dev)
print(c2_dev_max)

#Q7

Ac1 = np.dot(A,c1)
Ac2 = np.dot(A,c2)

x=np.linspace(0,2*(np.pi),len(Ac1))
xlabel('$x\longrightarrow$')
ylabel('$exp(x),log scale\longrightarrow$')
title('Plot of Ac1 and exponent with logy scale')
yscale('log')
scatter(x , Ac1, color='green', marker='o', label = 'Ac1 vs actual plot')
x = np.arange(-2*(np.pi),4*(np.pi),0.1)
plot(x , y1, color = 'blue', label = 'exp(x)')
plot(x , y3, color = 'black', label = 'exp(x)_periodic')
legend(loc = 'upper right')
grid()
show()

x=np.linspace(0,2*(np.pi),len(Ac2))
xlabel('$x\longrightarrow$')
ylabel('$cos(cos(x))\longrightarrow$')
title('Plot of Ac2 and cos of cos')
scatter(x , Ac2, color='green', marker='o', label = 'Ac2 vs actual plot')
x = np.arange(-2*(np.pi),4*(np.pi),0.1)
plot(x , y2, color = 'red', label = 'cos(cos(x)')
legend(loc = 'upper right')
grid()
show()