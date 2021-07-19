
### EE2703 Endsem
### EE19B070 , Anvith Pabba

import numpy as np
import pylab
import matplotlib.pyplot as plt

#Start of the Code

no_of_sections = 100
theeta = np.arange(0,2*np.pi,2*np.pi/no_of_sections) #diving the circle into 100 sections
a = 10 #radius of the loop


#x and y components of the radial vector, r'l bar
rx = a*np.cos(theeta) #x component
ry = a*np.sin(theeta) #y component


#x and y components of the current carrying element at an angle of theeta
Ix = -1*(2*np.pi/(4*np.pi*1e-7))*np.sin(theeta)*np.cos(theeta)
Iy = (2*np.pi/(4*np.pi*1e-7))*np.cos(theeta)*np.cos(theeta)


#plotting the scatter plot of all the current carrying elements in the loop
plt.scatter(rx,ry, c = 'red', s = 8, label = 'Current Elements')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Centre Points Of The Current Elements')
plt.xlim((-12,12))
plt.ylim((-12,12))
plt.legend(loc = 'upper right')
plt.show()


#plotting the quiver plot of currents in the current carrying elements
plt.quiver(rx,ry,Ix,Iy, color = 'black',label = 'Current Vectors')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Quiver Plot Of Currents')
plt.xlim((-12,12))
plt.ylim((-12,12))
plt.legend(loc = 'upper right')
plt.show()


#x and y components of dl'; the length vector of each incremental element
dlx = (2*np.pi*a/no_of_sections)*(-1*np.sin(theeta)) 
dly = (2*np.pi*a/no_of_sections)*(np.cos(theeta))


#creating the mesh grid
x = np.linspace(0,2,3) # breaking the volume into 3x3x1000
x = x-1  #since we want to make a 3x3 grid in the x-y plane that is from [-1,0,1] NOT [0,1,2]
y = np.linspace(0,2,3)
y = y-1
z = np.linspace(1,1000,1000)
xx,yy,zz = np.meshgrid(x,y,z) 
rijk = np.array((xx,yy,zz))


##rijk.shape

#rijk[1,0,2,900] 
#this gives us a 4d vector rijk in which if we set the FIRST parameter to 0 then it will give the y-coordinate
#of the point rijk, similarly if set it as 1 then it gives the x coordinate

#hence, we switch both of these for simpler use in a variable rijk_:

rijk_ = np.zeros((3,3,3,1000))
rijk_[0,:,:,:] = rijk[1,:,:,:]
rijk_[1,:,:,:] = rijk[0,:,:,:]
rijk_[2,:,:,:] = rijk[2,:,:,:]
rijk_[0,1,2,900] #ouput has to be the x coord of point with[1,2,900] which is 0

#this gives us a 4d vector rijk_ in which if we set the FIRST parameter to 0 then it will give the x-coordinate
#of the point rijk, similarly if set it as 1 then it gives the y coordinate

'''
now we define a function calc
where, the input is 'l'
and the output is a vector with all the magnitude of distances between each point in the volume and the lth point on the loop
'''
def calc(l):
    Rijk = np.zeros((1,3,3,1000))
    Rijk[0,:,:,:] = ((rijk_[0,:,:,:]-rx[l])**2 + (rijk_[1,:,:,:]-ry[l])**2 + (rijk_[2,:,:,:])**2 )**(1/2)
    return Rijk
    
#example = calc(0)
#example[0,1,1,999]   ;output is 1000.049.. which is the distance between [10,0,0] and [0,0,1000]
#TEST WAS SUCCESSFULLL!!!!
    

def A_x_l(l): #now we define a function that gives us the x component of A at all points from a the lth point on the loop
    A_x_ = np.zeros((1,3,3,1000))
    A_x_ = ( np.cos(theeta[l])* np.exp(-1*1j*0.1*calc(l))* dlx[l] )/ (calc(l))
    return A_x_

def A_y_l(l): #now we define a function that gives us the y component of A at all points from a the lth point on the loop
    A_y_ = np.zeros((1,3,3,1000))
    A_y_ = ( np.cos(theeta[l])* np.exp(-1*1j*0.1*calc(l))* dly[l] )/ (calc(l))
    return A_y_

#Finally, getting the x and y components of A through a summation
A_x = np.zeros((1,3,3,1000))
for l in range(100): #we us a for loop as we need to find a summantion through iterating 'l'
    A_x = A_x + A_x_l(l)

#The reason we use a for loop instead of vectorised code is because to make this operation possible in vectorised code,
#we would have to extend the A_x and A_y vector with a 5th parameter that has 100 terms (to match the 'l' parameter). then we could use vectorisation.
#this would us up alot of space even though it is faster, hence using a for loop is justified here.

A_y = np.zeros((1,3,3,1000))
for l in range(100):
    A_y = A_y + A_y_l(l) #see reason above


#finding the final B output
B = np.zeros(1000)
B = (A_y[0,1,0,:] - A_x[0,0,1,:] - A_y[0,-1,0,:] + A_x[0,0,-1,:])/(4) #delta x = delta y = 1

#loglog plot of B vs z
z = np.arange(1,1001,1)
plt.loglog(z,np.abs(B),label = 'Magnetic Flux Density')
plt.xlabel('Log(z)')
plt.ylabel('Log(Bz)')
plt.title('LogLog Plot Of Magnetic Flux Density In The z-Direction')
plt.legend(loc = 'upper right')
plt.show()

# now to fit the data into B = c*z^b
# therefore, log(B) = b*log(z) + c


#in this, we only fit the values for the samples after z=100, this is because the loglog plot is linear here
A_fit = np.c_[np.log(z[100:]),np.ones(len(B)-100)]
B_fit = np.log(abs(B)[100:])
a_real,b_real = np.linalg.lstsq(A_fit, B_fit, rcond=None)[0]
print ("The fitted b,c only taking samples after the graph turns linear is:\nb = {}, c = {}\n" .format(a_real,b_real))


#in this, we find a fit for all smaples
A_fit = np.c_[np.log(z),np.ones(len(B))]
B_fit = np.log(abs(B))
a_no_approx,b_no_approx = np.linalg.lstsq(A_fit, B_fit, rcond=None)[0]
print ("The fitted b,c taking all the samples into consideration is:\nb = {}, c = {}" .format(a_no_approx,b_no_approx))