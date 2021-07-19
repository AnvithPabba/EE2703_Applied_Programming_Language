import scipy.signal as sp
import numpy as np
from numpy import *
from pylab import *
import math 

#The Assignment:
#Question 1:


def H_s(a,b): #defining the Laplace trasnform, -a is the decay coefficiant, and b is the frequency
	num = poly1d([1,a])
	den1 = poly1d([1,2*a,a*a + b*b])
	den2 = poly1d([1,0,2.25])
	den = polymul(den1,den2)
	H = sp.lti(num,den)
	return H

H1 = H_s(0.5,1.5)

t,x = sp.impulse(H1,None, linspace(0,50,10000))
plot(t,x)
xlabel('t')
ylabel('x(t)')
title('outfput for decay coeff= -0.5 and freq= 1.5')
grid()
show()

#Question 2:

H2 = H_s(0.05,1.5)

t2,x2 = sp.impulse(H2,None, linspace(0,50,10000))
plot(t2,x2)
xlabel('t')
ylabel('x(t)')
title('outfput for decay coeff= -0.05 and freq= 1.5')
grid()
show()

#Question 3:

for i in range(5): #finding output for a range of frequencies from 1.4 to 1.6 with 0.05 increments
	f = 1.4 + i*0.05
	H = H_s(0.05,f)
	t,x = sp.impulse(H,None, linspace(0,100,10000))
	xlabel('t')
	ylabel('x')
	title('system response of spring with freq = %0.2f'%f)
	plot(t,x)
	grid()
	show()

#Question 4:

H_x = sp.lti([1,0,2],[1,0,3,0])
H_y = sp.lti([2],[1,0,3,0])

t,x = sp.impulse(H_x,None, linspace(0,20,10000))
xlabel('t')
ylabel('x')
title('system response of x')
plot(t,x)
grid()
show()

t,x = sp.impulse(H_y,None, linspace(0,20,10000))
xlabel('t')
ylabel('x')
title('system response of y')
plot(t,x)
grid()
show()

#Question 5:

L = 1e-6
C = 1e-6
R = 100

H = sp.lti(1,[L*C,R*C,1]) #transfer function
w,S,phi = H.bode()
semilogx(w,S)
xlabel('log(w)')
ylabel('log(|H(jw)|')
title('bode magnitude response')
plot(w,S)
grid()
show()

semilogx(w,phi)
xlabel('log(w)')
ylabel('phase(|H(jw)|')
title('bode phase response')
plot(w,phi)
grid()
show()

#Question 6:

t = linspace(0,1e-2,10000)
v_i = cos(1e3*t) - cos(1e6*t)
t,y,svec = sp.lsim(H,v_i,t)
plot(t,y)
xlabel('t')
ylabel('$H * v_i = v_o$')
title('output response using convolution upto t=10ms')
grid()
show()

t = linspace(0,30*1e-6,10000)
v_i = cos(1e3*t) - cos(1e6*t)
t,y,svec = sp.lsim(H,v_i,t)
plot(t,y)
xlabel('t')
ylabel('$H * v_i = v_o$')
title('output response using convolution upto t = 30\u03BCs')
grid()
show()


t = linspace(3*1e-3,3.25*1e-3,10000)
v_i = cos(1e3*t) - cos(1e6*t)
t,y,svec = sp.lsim(H,v_i,t)
plot(t,y)
xlabel('t')
ylabel('$H * v_i = v_o$')
title('Fast twitching in output response')
grid()
show()