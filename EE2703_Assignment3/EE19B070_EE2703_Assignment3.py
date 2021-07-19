import numpy as np
from numpy import *
import scipy
from scipy.special import *
from pylab import *
import math

sigma = logspace(-1,-3,9) #generates an array from 0.1 to 0.001 with 9 terms in it

#unpacking the file data into "a"
data = loadtxt('fitting.dat', dtype='float', comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=True, ndmin=0)

#Subscripts for numbers
subscripts = ['\u2081','\u2082','\u2083','\u2084','\u2085','\u2086','\u2087','\u2088','\u2089']

#Defining bessel Function
def g(t, A, B):
    return A*jn(2,t)+B*t

#print(str(np.round(sigma[5],4)))
#plotting all 9 graphs with 9 different noise values/standard deviations
ylabel('$f(t)+noise$',size = 15)
xlabel('$t$',size = 15)
title('Figure 0')
for k in range(9):
	labels = '\u03C3' + subscripts[k] + '=' + str(np.round(sigma[k],3))
	plot(data[0],data[k+1],label = labels)
	legend(loc = 'upper right')
plot(data[0],g(data[0],1.05,-0.105),color = 'black', label = 'f(t)')
legend(loc = 'upper right')
grid(True)
show()


#plotting the errorbar of noise vs time
xlabel('$t$',size = 15)
title('Q5: Data points for \u03C3=0.100 along with the exact function')
errorbar(data[0][::5],data[1][::5],sigma[0],fmt='ro',label='Errorbar')  #errorbar plot, standard deviation is 0.1 (== sigma[0])
plot(data[0],g(data[0],1.05,-0.105),label = 'f(t)', color= 'black' )
legend(loc = 'upper right')
grid(True)
show()

#to check the equivalence of g(t,A,b) and M.p

J = []
for t in data[0]:
	J.append(jn(2,t))
M = c_[J,data[0]] #LHS M array

A0 = 1.05
B0 = -0.105
p = [[A0],[B0]]

print('\nChecking to see if g(t,A,B) == M.p:')
RHS = c_[g(data[0], A0, B0)]
LHS = dot(M,p)

if array_equal(LHS,RHS):
	print('g(t,A0,B0) is EQUAL to M.p\n')
else:
	print('g(t,A0,B0) is NOT EQUAL to M.p\n')

#to plot the mean square error for various values of A and B	
A = zeros(21)
B = zeros(21)
for a in range(21):
	A[a] = 0.1*a
for b in range(21):
	B[b] = -0.2 + 0.01*b
e = zeros((21,21), dtype = float)

#print(A)
#print(A)

for i in range(21):
	for j in range(21):
		for k in range(len(data[0])):
			e[i][j] += (1/101)*((data[1][k]-g(data[0][k],A[i],B[j]))**2)

#to find the minimas
error_min = 100
for i in range(21):
	for j in range(21):
		if e[i][j] < error_min:
			error_min = e[i][j]

#compare = error_min - amin(e)  #test to see if above code is correct
#print(compare)

num_min = 0
for i in range(21):
	for j in range(21):
		if e[i][j] == error_min:
			print('the minimum error is : {}'.format(error_min))
			num_min += 1
print('the number of minimums is : {}'.format(num_min))

#Plotting the contour plot

xlabel('$A$ $\longrightarrow$')
ylabel('$B% $\longrightarrow$')
title('Q8: Contour plot of \u03B5\u1d62\u2c7c')
c_graph = contour(A,B,e,20)
clabel(c_graph,c_graph.levels[:5],inline=True)
plot(A0,B0,'ro')
text(A0,B0,'Exact location',color='red')   #putting text on graph
grid(True)
show()

#obtaining the best estimate of A and B
Aerr = []
Berr = []

#print(len(data))
for i in range(len(data)-1):
	dA,dB = np.linalg.lstsq(M,data[i+1],rcond=None)[0]
	Aerr.append(abs(dA - A0))
	Berr.append(abs(dB - B0))

ylabel('$MS error \longrightarrow$')
xlabel('$Noise Standard Deviation $\longrightarrow$')
title('Q10: Variation of error with noise')
plot(sigma,Aerr,linestyle='dashed',color='red',marker='o',mfc='red',mec='red',label='Aerr')
plot(sigma,Berr,linestyle='dashed',color='green',marker='o',mfc='green',mec='green',label='Berr')
legend()
grid(True)
show()

#plotting the variation of Aerr an Berr with respect to noise in the log scale
errorbar(sigma,Aerr,sigma,fmt='ro',label='Aerr')
errorbar(sigma,Berr,sigma,fmt='go',label='Berr')
#cnverting the axis scale to log
yscale(value='log')
xscale(value='log')
ylabel('$MS Error$ $\longrightarrow$')
xlabel('\u03C3\u2099',size=15)
title('Q11: Variation of error with noise')
legend()
grid(True)
show()





