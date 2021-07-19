import sys
import numpy
from numpy import *
import matplotlib
from matplotlib.pyplot import *

#We are supposed to get 5 inputs

#n=100 # spatial grid size.
#M=5 # number of electrons injected per turn.
#nk=500 # number of turns to simulate.
#u0=5 # threshold velocity.
#p=0.25 # probability that ionization will occur

if len(sys.argv) != 6:
	print("There has to be 6 integer inputs")

inp = sys.argv
input_int = ['']*(5)

#checks if first 3 inputs are integera
for n in range(3):
	try:
		input_int[n] = int(sys.argv[n+1])
	except ValueError:
	    print("Some of the first 3 inputs are NOT an integer, please check!!")
	    break
	else:
		input_int[n] = int(sys.argv[n+1])

#checks if last 2 inputs are floating integers
for n in range(2):
	try:
		input_int[n+3] = float(sys.argv[n+4])
	except ValueError:
	    print("Some of the last 2 inputs are NOT a rational number, please check!!")
	    break
	else:
		input_int[n+3] = float(sys.argv[n+4])

#assigning the values

n = input_int[0]
M = input_int[1]
nk = input_int[2]
u0 = input_int[3]
p = input_int[4]

#print(input_int)

xx = zeros(n*M)
u = zeros(n*M)
dx = zeros(n*M)

I = []
X = []
V = []


for i in range(1,nk):
	ii = where(xx > 0)[0]
	dx[ii] = u[ii] + 0.5
	xx[ii] = xx[ii] + dx[ii]
	u[ii] = u[ii] + 1

	out_of_bounds = where(xx > n)[0]
	xx[out_of_bounds] = 0
	u[out_of_bounds] = 0
	dx[out_of_bounds] = 0

	#defing the deviation of electrons added per turn
	Msig = 2


	kk = where(u > u0)
	ll=where(random.randn(len(kk[0]))<=p)[0];
	kl=kk[0][ll];

	xx[kl] = xx[kl] - dx[kl]*random.rand(len(kl))

	I.extend(xx[kl].tolist())
	m=int(random.randn()*Msig+M)

	empty = where(xx == 0)[0]
	tt = min(len(empty),m)

	xx[empty[:tt]] = 1
	u[empty[:tt]] = 0
	dx[empty[:tt]] = 0


	ii = where(xx > 0)[0]
	X.extend(xx[ii].tolist())
	V.extend(u[ii].tolist())


#plot the number of electrons vs x
hist(X, bins = arange(0,101), rwidth=0.75, color = 'purple')
title('Number of Electrons vs $x$ with $u_0=$%0.3f and p=%0.3f'%(u0,p))
xlabel('$x$')
ylabel('Number of electrons')
show()

#plot the Intensity vs x
hist(I, bins = arange(0,101), rwidth=0.75, color = 'red')
title('Intensity vs $x$ with $u_0=$%0.3f and p=%0.3f'%(u0,p))
xlabel('$x$')
ylabel('Intensity')
show()

#electron phase plot
plot(X,V,marker ='x', color = 'yellow')
title('Electron phase plot')
xlabel('$x$')
ylabel('$v$')
show()
