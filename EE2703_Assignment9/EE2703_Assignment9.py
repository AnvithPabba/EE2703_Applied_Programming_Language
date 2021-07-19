#EE2703
#Anvith Pabba [EE19B970]
#Assignment 9 - DFT of aperiodic signals

import numpy
from numpy import *
import pylab
from pylab import *
import random
from random import *
from mpl_toolkits.mplot3d import Axes3D


#DFT of sin(sqrt(2)t)
t=linspace(-pi,pi,65);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
y=sin(sqrt(2)*t)
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/64.0
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]

subplot(2,1,1)
plot(w,abs(Y),w,abs(Y),'bo',lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
grid(True)

subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()


#plots of sin(sqrt(2)t) for different time ranges
t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]

# y=sin(sqrt(2)*t)
plot(t1,sin(sqrt(2)*t1),'b',lw=4)
plot(t1+2*pi,sin(sqrt(2)*t1),'purple',lw=2)
plot(t1-2*pi,sin(sqrt(2)*t1),'purple',lw=2)
plot(t2,sin(sqrt(2)*t2),'r',lw=4)
plot(t3,sin(sqrt(2)*t3),'r',lw=4)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)$")
grid(True)
show()



#plot of sin(sqrt(2)t) after windowing:
t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]
n=arange(64)
wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
y=sin(sqrt(2)*t1)*wnd
figure(3)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'ro',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
grid(True)
show()



#DFT of the function after windowing

t=linspace(-pi,pi,65);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(64)
wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
y=sin(sqrt(2)*t)*wnd
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/64.0
w=linspace(-pi*fmax,pi*fmax,65);w=w[:-1]

subplot(2,1,1)
plot(w,abs(Y),w,abs(Y),'bo',lw=2)
xlim([-8,8])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$ with 64 points")
grid(True)

subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-8,8])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

#DFT after windowing with 256 points


t=linspace(-4*pi,4*pi,257);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(256)
wnd=fftshift(0.54+0.46*cos(2*pi*n/256))
y=sin(sqrt(2)*t)
# y=sin(1.25*t)
y=y*wnd
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/256.0
w=linspace(-pi*fmax,pi*fmax,257);w=w[:-1]

subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times$ w(t) with 256 points")
grid(True)

subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()


#Question 2:

#DFT of cos^3(wot) where wo=0.86 without hamming window

t=linspace(-pi,pi,257);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
wo=0.86
y=(cos(wo*t))**3
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/256.0
w=linspace(-pi*fmax,pi*fmax,257);w=w[:-1]

subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos^3(w_ot)$")
grid(True)


subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()



#DFT of cos^3(wot) where wo=0.86 with the hamming window


t=linspace(-4*pi,4*pi,257);t=t[:-1]
dt=t[1]-t[0];fmax=1/dt
n=arange(256)
wnd=fftshift(0.54+0.46*cos(2*pi*n/256))
wo=0.86
y=(cos(wo*t))**3
# y=cos^3(wo*t)
y=y*wnd
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/256.0
w=linspace(-pi*fmax,pi*fmax,257);w=w[:-1]


subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos^3(w_ot)\times w(t)$")
grid(True)


subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()



#Question 3: finding wo and delta from a randomly generated vector

N=128

wo = random() + 0.5
delta = random()*2*pi - pi
t=linspace(-8*pi,8*pi,N+1);t=t[:-1] #time period of -pi to pi has very low precision
dt = t[1]-t[0];fmax=1/dt
n = arange(N)
wnd = fftshift(0.54+0.46*cos(2*pi*n/N))
vec = cos(wo*t + delta)
vec = vec*wnd
vec[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(vec) # make y start with y(t=0)
Y=fftshift(fft(y))/N
w=linspace(-pi*fmax,pi*fmax,N+1);w=w[:-1]
print('Actual wo and delta is: %f,%f'%(wo,delta))

subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
ylabel(r"$|Y|$",size=16)
xlim([-4,4])
title(r"Spectrum of $cos(wot+\delta)$")
grid(True)

subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
xlim([-4,4])
grid(True)
show()

max = abs(Y).max()
ii = where(abs(abs(Y)-(max))<0.01)
ii2 = where(abs(Y)>0)
wapprox = sum(abs(Y[ii2])*abs(Y[ii2])*abs(w[ii2])) / sum(abs(Y[ii2])*abs(Y[ii2]))
print('The angular frequency wo is approximately:',wapprox)
print('The phase shift delta is approximately:',angle(Y[ii][1]))


#Question 4: Same as above, but now we find it for a noisy signal

N=256

wo = random() + 0.5
delta = random()*2*pi - pi
t=linspace(-8*pi,8*pi,N+1);t=t[:-1] #time period of -pi to pi has very low precision
dt = t[1]-t[0];fmax=1/dt
vec = cos(wo*t + delta)
vec = vec + 0.1*randn(N)
vec[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(vec) # make y start with y(t=0)
Y=fftshift(fft(y))/N
w=linspace(-pi*fmax,pi*fmax,N+1);w=w[:-1]
print('\nActual wo and delta of the noisy function is: %f,%f'%(wo,delta))

subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos(wot+\delta)$")
grid(True)


subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

max = abs(Y).max()
ii = where(abs(abs(Y)-(max))<0.01)
ii2 = where(abs(Y)>0.15)
wapprox = sum(abs(Y[ii2])**1.6*abs(w[ii2])) / sum(abs(Y[ii2])**1.6)
print('The angular frequency wo of the noisy function is approximately:',wapprox)
print('The phase shift delta of the noisy function is approximately:',angle(Y[ii][1]))


#Question 5: DFT of a 'chirped' signal

N=1024

t=linspace(-pi,pi,N+1);t=t[:-1] #time period of -pi to pi has very low precision
dt = t[1]-t[0];fmax=1/dt
y = cos(16*t*(1.5 + (t/(2*pi))))
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/N
w=linspace(-pi*fmax,pi*fmax,N+1);w=w[:-1]

subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-50,50])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos(16(1.5+t/2\pi)t)$")
grid(True)


subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-50,50])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

#with hamming window
N=1024

t=linspace(-pi,pi,N+1);t=t[:-1] #time period of -pi to pi has very low precision
dt = t[1]-t[0];fmax=1/dt
n = arange(N)
wnd = fftshift(0.54+0.46*cos(2*pi*n/N))
y = cos(16*t*(1.5 + (t/(2*pi))))
y = y*wnd
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y=fftshift(y) # make y start with y(t=0)
Y=fftshift(fft(y))/N
w=linspace(-pi*fmax,pi*fmax,N+1);w=w[:-1]

subplot(2,1,1)
plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
xlim([-50,50])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos(16(1.5+t/2\pi)t)$ with hamming window")
grid(True)


subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-50,50])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

#Question 6: Surface Plot of Variation of frequency with time for both types of chirped signals
#Without hamming window:

N=1024

t=linspace(-pi,pi,N+1);t=t[:-1] #time period of -pi to pi has very low precision
dt = t[1]-t[0];fmax=1/dt
y = cos(16*t*(1.5 + (t/(2*pi))))

subvec = zeros((16,64))
for n in arange(16):
    subvec[n]=y[64*n:64*n + 64]
    subvec[n][0] = 0

y=fftshift(subvec) # make y start with y(t=0)
Y=fftshift(fft(y))/N
w=linspace(-pi*fmax,pi*fmax,N+1);w=w[:-1]
n=arange(64)


t1 = np.array ( arange (16) )
t1,n = meshgrid ( t1 , n )
ax = Axes3D(figure())
surf = ax.plot_surface ( t1 ,n ,abs(Y).T , rstride =1 , cstride =1 , cmap ='inferno')
ylabel('\u03C9')
xlabel('t')
title(" Surface Plot of Variation of frequency with time - Chirped Signal without Hamming Window ")
ax.set_zlabel ('|Y|')
show()


#With hamming window:
N=1024

t=linspace(-pi,pi,N+1);t=t[:-1] #time period of -pi to pi has very low precision
dt = t[1]-t[0];fmax=1/dt
n = arange(N)
wnd = fftshift(0.54+0.46*cos(2*pi*n/N))
y = cos(16*t*(1.5 + (t/(2*pi))))
y = y*wnd

subvec = zeros((16,64))
for n in arange(16):
    subvec[n]=y[64*n:64*n + 64]
    subvec[n][0] = 0


y=fftshift(subvec) # make y start with y(t=0)
Y=fftshift(fft(y))/N
w=linspace(-pi*fmax,pi*fmax,N+1);w=w[:-1]
n=arange(64)


t1 = np.array ( arange (16) )
t1,n = meshgrid ( t1 , n )
ax = Axes3D( figure())
surf = ax.plot_surface ( t1 ,n ,abs(Y).T , rstride =1 , cstride =1 , cmap ='inferno')
ylabel('\u03C9')
xlabel('t')
title(" Surface Plot of Variation of frequency with time - Chirped Signal with Hamming Window ")
ax.set_zlabel ('|Y|')
show()



#end