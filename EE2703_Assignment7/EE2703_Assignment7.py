import sympy
from sympy import *
import scipy.signal as sp
import pylab as p 

init_session #necessar when we use sympy, == initialise session

#defining a plotter function
def plot_display(x,y,xlabel,ylabel,title,xscale,yscale):
	p.plot(x,y)
	p.xlabel(xlabel)
	p.ylabel(xlabel)
	p.xscale(xscale)
	p.yscale(yscale)
	p.title(title)
	p.grid()
	p.show()

#converts an equation containing sympy variables into a function that scipy accepts
def sympy_to_H_lti(sympy_func,s):
	s=symbols('s')
	n,d = fraction(sympy_func)	#getting the numerator and denominator
	num_c = Poly(n,s).all_coeffs()	#getting the coeff
	den_c = Poly(d,s).all_coeffs()
	numf = p.array(num_c, dtype=float)	#converting the arrays into numpy arrays
	denf = p.array(den_c, dtype=float)
	H = sp.lti(numf,denf)	#creating the lti systems transfer response
	return(H)



#Defining a Low pass filter in sympy with necessary parameters
def lowpass(R1,R2,C1,C2,G,Vi):
	s=symbols('s')
	# Creating the matrices through nodal analysis then solvng them to get the output
	A=Matrix([[0,0,1,-1/G], [-1/(1+s*R2*C2),1,0,0], [0,-G,G,1], [-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
	b=Matrix([0,0,0,-Vi/R1])
	V=A.inv()*b
	return(A,b,V)

#using sympy to define s as a symbol
s=symbols('s')

#finding the solutoion matrix
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
Vo=V[3]
w=p.logspace(0,8,801)
ss=1j*w
hf=lambdify(s,Vo,'numpy')
v=hf(ss)

#plotting the magnitude plot of the low pass filter
plot_display(w,abs(v),'log(w)','log(|H(jw)|)','bode magnitude plot of the low pass function','log','log')

#Assignment

#Question 1:

#step impulse response, Vi(s) = 1/s
A,b,V1=lowpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo1 = V1[3]
w = p.logspace(0,8,801)
ss=1j*w
hf1=lambdify(s,Vo1,'numpy')
v=hf1(ss)
H1 = sympy_to_H_lti(Vo1,s)

#plotting the magnitude plot of the step response of the low pass filter
plot_display(w,abs(v),'w','magnitude','Magnitude bode plot of step response of low pass filter','log','log')

t,y=sp.impulse(H1,None,p.linspace(0,0.001,1000))
plot_display(t,y,'t','y','step response of low pass filter','linear','linear')

#Question 2:

#getting the input signal
t = p.linspace(0,0.005,10000)
v_i = p.multiply(p.sin(2000*p.pi*t) + p.cos(2*1e6*p.pi*t),p.heaviside(t,0.5))

#getting the transfer function of the LPF
H2 = sympy_to_H_lti(Vo,s)

#performing the convolution
t,y,svec=sp.lsim(H2,v_i,t)
plot_display(t,y,'t','y','output response of input sinusoids','linear','linear')

#Question 3:

#Defining a High pass filter in sympy with necessary parameters

def highpass(R1,R3,C1,C2,G,Vi):
    s = symbols("s")
	# Creating the matrices through nodal analysis then solvng them to get the output
    A = Matrix([[0,-1,0,1/G],[s*C2*R3/(s*C2*R3+1),0,-1,0],[0,G,-G,1],[-s*C2-1/R1-s*C1,0,s*C2,1/R1]])
    b = Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return A,b,V

A3,b3,V3 = highpass(10000,10000,1e-9,1e-9,1.586,1)
Vo3 = V3[3]
H3 = sympy_to_H_lti(Vo3,s)
w = p.logspace(0,8,801)
ss=1j*w
hf3 = lambdify(s,Vo3,'numpy')
v3 = hf3(ss)

#bode magnitude plot of a high pass active filter
plot_display(w,abs(v3),'log(w)','|H3(jw)|','Bode magnitude plot of high pass','log','log')

#Question 4:

#defining a function that gives the output as damped sinusoid
#with all the necessary input parameters
def damped_sinusoid(freq,type,decay,t):
	if type == 'sin':
		return p.sin(freq*2*p.pi*t)*p.exp(-1*decay*t)
	if type == 'cos':
		return p.cos(freq*2*p.pi*t)*p.exp(-1*decay*t)


#plotting the input and output graphs for a low freq, high decay sinusoid
t = p.linspace(0,0.01,10000)
v_i3 = damped_sinusoid(2000,'sin',1000,t)
t,y,svec=sp.lsim(H3,v_i3,t)
plot_display(t,damped_sinusoid(2000,'sin',1000,t),'t','$v_i3$','damped sinusoid input response (low freq)','linear','linear')
plot_display(t,y,'t','y','output response','linear','linear')


#plotting the input and output graphs for a high freq, high decay sinusoid
t = p.linspace(0,0.01,10000)
v_i4 = damped_sinusoid(20000,'sin',1000,t)
t,y,svec=sp.lsim(H3,v_i4,t)
plot_display(t,damped_sinusoid(20000,'sin',1000,t),'t','$v_i4$','damped sinusoid input response (high freq)','linear','linear')
plot_display(t,y,'t','y','output response','linear','linear')

#Question 5:

#input is Vi(s) = 1/s
#we find the output of the filter for this kind of response
A5,b5,V5 = highpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo5 = V5[3]
H5 = sympy_to_H_lti(Vo5,s)
w = p.logspace(0,8,801)
ss=1j*w
hf5 = lambdify(s,Vo5,'numpy')
v5 = hf5(ss)

#bode magnitude plot of a high pass active filter
plot_display(w,abs(v5),'log(w)','|H5(jw)|','Bode magnitude plot of step response of high pass filter','log','log')

#impulse step response of the high pass filter
t = p.linspace(0,0.01,1000)
t,y=sp.impulse(H5,None,p.linspace(0,0.001,1000))
plot_display(t,y,'t','y','step response','linear','linear')

