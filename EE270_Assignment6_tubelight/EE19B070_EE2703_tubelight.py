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

#checks if first 3 inputs are integers
for n in range(3):
	try:
		input_int[n] = int(sys.argv[n+1]) #checking if converting to int throws us an error
	except ValueError:
	    print("Some of the first 3 inputs are NOT an integer, please check!!")
	    break
	else:
		input_int[n] = int(sys.argv[n+1])

#checks if last 2 inputs are floating integers
for n in range(2):
	try:
		input_int[n+3] = float(sys.argv[n+4]) #checking if converting to float throws us an error
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


#initialising the vectors
I = []
X = []
V = []

#begin for loop
for i in range(1,nk): #iterate the block nk times
	ii = where(xx > 0)[0] 	#finds location of all existing electrons
	dx[ii] = u[ii] + 0.5	#gives us change in location
	xx[ii] = xx[ii] + dx[ii]	#adding this change
	u[ii] = u[ii] + 1			#change in velcity due to the constant acceleration

	out_of_bounds = where(xx > n)[0]	#finds location of all electrons out of bounds
	xx[out_of_bounds] = 0				#initialising all the values of these electrons to 0
	u[out_of_bounds] = 0
	dx[out_of_bounds] = 0

	#defing the deviation of electrons added per turn
	Msig = 2

	#for the collisions
	kk = where(u > u0)	#finds location of all electrons that can cause collisions
	ll=where(random.rand(len(kk[0]))<=p)[0];	#random distrinution of electrons that collide
	kl=kk[0][ll];								

	u[kl] = 0	#velocity of electrons after collision is 0
	xx[kl] = xx[kl] - dx[kl]*random.rand(len(kl))	#new electron position

	I.extend(xx[kl].tolist())	#adding all of these elctrons to our I vector
	m=int(random.rand()*Msig+M)	#number of electrons added is 5 with a standard deviation of 2

	empty = where(xx == 0)[0]
	tt = min(len(empty),m)		#checking which one is smaller

	xx[empty[:tt]] = 1	#setting the position of these elecrons to 1 and the rest of their values to 0
	u[empty[:tt]] = 0
	dx[empty[:tt]] = 0


	ii = where(xx > 0)[0]	#again we find existing electrons and add it to X and V
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
plot(X,V,'bx')
title('Electron phase plot with $u_0=$%0.3f and p=%0.3f'%(u0,p))
xlabel('$x$')
ylabel('$v$')
show()

#plots the Intensity density plot
bins = arange(0,101)
hist, edges = np.histogram(I, bins)
hist=hist[newaxis,:]
extent=[bins.min(), bins.max(),0,1]
imshow(hist, aspect = "auto", cmap="Greys_r", extent=extent)
title('Intensity Density plot with $u_0=$%0.3f and p=%0.3f'%(u0,p))
gca().set_yticks([])
colorbar()
show()

#printing the table
print('\nIntensity Data Table:\n')
print('___________________________________\n')
print('xpos			count\n')
print('________________|__________________\n')
for k in range(100):
	print('%f	|	%f'%(bins[k],hist[0][k]))
print('___________________________________\n')


