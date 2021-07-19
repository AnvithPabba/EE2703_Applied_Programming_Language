import numpy as np
from numpy import *
import scipy
from scipy.integrate import *
import math
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3

import sys

#Checking if correct number of arguments are given
if len(sys.argv)!=5:
	print("Please enter ALL 4 parameters!!")

#Checking if all the inputs are intgers
while True:
	try:
		Nx = int(sys.argv[1])       
	except ValueError:
		print('argument 1 must be an integer')
		break
	else:
		Nx = int(sys.argv[1])
		break 

while True:
	try:
		Ny = int(sys.argv[2])       
	except ValueError:
		print('argument 2 must be an integer')
		break
	else:
		Nx = int(sys.argv[2])
		break 

while True:
	try:
		r = int(sys.argv[3])       
	except ValueError:
		print('radius must be an integer')
		break
	else:
		r = int(sys.argv[3])
		break 

while True:
	try:
		Niter = int(sys.argv[4])       
	except ValueError:
		print('Number of iterations must be an integer')
		break
	else:
		Niter = int(sys.argv[4])
		break 

#start

x = np.linspace(-((Nx-1)/2), ((Nx-1)/2), Nx)
y = np.linspace(-((Ny-1)/2), ((Ny-1)/2), Ny) 

#initialising phi
phi = np.zeros([Ny,Nx])
X,Y=meshgrid(x,y)
n = arange(Niter)
errors = np.zeros([Niter,1])

ii = where((X*X + Y*Y) <= r*r)
phi[ii] = 1.0

#Figure 1
# Plotting the potential with only the initial V = 1 condition
plot(ii[0] - (Nx-1)/2, ii[1] - (Ny-1)/2,linestyle="None", color='red', marker='o',label="V = 1")
title("Figure(1) - Initial Potential Contour")
xlabel('$X\longrightarrow$')
ylabel('$Y\longrightarrow$')
xticks(arange(-12,13,2))
yticks(arange(-12,13,2))
xlim(- (Nx-1)/2 , (Nx-1)/2 )
ylim(- (Ny-1)/2 , (Ny-1)/2 )
grid(True)
legend()
show()


#perform the iteration
oldphi=phi.copy()


for k in range(Niter):


	oldphi=phi.copy()


	#update phi array
	phi[1:-1,1:-1] = 0.25*(phi[1:-1,:-2] + phi[1:-1,2:] + phi[:-2,1:-1] + phi[2:,1:-1])



	#assert boundaries
	phi[1:-1,0] = phi[1:-1,1]	#left side
	phi[1:-1,-1] = phi[1:-1,-2]	#right side
	phi[0,1:-1] = phi[1,1:-1]	#top side
	#bottom is fixed with a value of 0

	phi[ii] = 1.0


	#error calculation
	errors[k]=(abs(phi-oldphi)).max();



fit1 = zeros([1000,1])
fit2 = zeros([1500,1])
#using lstsq
#lstsq for after 500
c1 = np.linalg.lstsq(c_[ones([1000,1]),n[500:1500]], log(errors[500:1500]),rcond = None)[0]
logA1 = c1[0]
A1 = exp(logA1)
B1 = c1[1]

#lstsq for the entire thing
c2 = np.linalg.lstsq(c_[ones([1500,1]),n], log(errors),rcond = None)[0]
logA2 = c2[0]
A2 = exp(logA2)
B2 = c2[1]

print("fitted values of A1 and B1 for iterations >500 are :",A1,B1)
print("fitted values of A2 and B2 are :",A2,B2)

Jx = np.zeros([Ny,Nx])
Jy = np.zeros([Ny,Nx])

#Jx,Jy for the quiver

Jx[1:-1, 1:-1] = 0.5*(phi[1:-1, 0:-2] - phi[1:-1, 2:])
Jy[1:-1, 1:-1] = 0.5*(phi[2:, 1:-1] - phi[0:-2, 1:-1])




title("Figure(2) - Error plot 1")
semilogy(n,errors,label="semilogy plot of error vs iteration")
xlabel('$n\longrightarrow$')
ylabel('$error,log[n]\longrightarrow$')
grid(True)
legend()
show()

title("Figure(3) - Error plot 2")
loglog(n,errors,label="loglog plot of error vs iteration")
xlabel('$n,log\longrightarrow$')
ylabel('$error[n],log\longrightarrow$')
grid(True)
legend()
show()

title("Figure(4) - Error plot 3, iterations after 500")
semilogy(n[500:1500],errors[500:1500],label="semilogy plot of error vs iteration")
xlabel('$n[1000values]\longrightarrow$')
ylabel('$error,log[n>500]\longrightarrow$')
grid(True)
legend()
show()

title("Figure(5) - Error plot 4, iterations after 500")
semilogy(n[500:1500],errors[500:1500],label="errors")
semilogy(n,A2*(exp(B2*n)),label="fit2")
semilogy(n[500:1500],A1*(exp(B1*n[500:1500])),label="fit1")
xlabel('$n\longrightarrow$')
ylabel('$error,log\longrightarrow$')
grid(True)
legend()
show()

title("Figure(6) - Error plot 5, Niter iterations")
semilogy(n,errors,label="errors")
semilogy(n,A2*(exp(B2*n)),label="fit2")
semilogy(n[500:1500],A1*(exp(B1*n[500:1500])),label="fit1")
xlabel('$n\longrightarrow$')
ylabel('$error,log\longrightarrow$')
grid(True)
legend()
show()

title("Figure(7) - Current quiver plot")
quiver(x,y,Jx[::-1,:],Jy[::-1,:],scale = 5,label = "current")
plot(ii[0] - (Nx-1)/2, ii[1] - (Ny-1)/2,linestyle="None", color='red', marker='o',label="V = 1")
xlabel('$x\longrightarrow$')
ylabel('$y\longrightarrow$')
grid(True)
legend(loc = 'upper right')
show()

title("Figure(8) - The 3-D surface plot of the potential")
fig1=figure(8) # open a new figure
ax=p3.Axes3D(fig1) # Axes3D is the means to do a surface plot
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=cm.jet)
ylabel('$Ground$')
xlabel('$y\longrightarrow$')
show()

title("Figure(9) - The contour plot")
contourf(x, -y, phi)
colorbar()
plot(ii[0] - (Nx-1)/2, ii[1] - (Ny-1)/2,linestyle="None", color='red', marker='o',label="V = 1")
xlabel("$x\longrightarrow$")
ylabel("$y\longrightarrow$")
show()