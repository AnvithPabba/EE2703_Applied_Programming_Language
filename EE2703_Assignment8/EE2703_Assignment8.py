import numpy
from numpy import *
import pylab
from pylab import *


#Q1: Working through the examples in the Assignment

#DFT of sin(5t)
x=linspace(0,2*pi,129);x=x[:-1]
y=sin(5*x)
Y=fftshift(fft(y))/128.0
w=linspace(-64,63,128)

#Magnitude plot of the DFT of sin(5t)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
xlabel(r"$w$",size=16)
title(r"Spectrum of $\sin(5t)$")
grid(True)
show()

#Phase plot of the DFT of sin(5t)
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)				#plots the points in green only where magnitude is greater than 0.001
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$w$",size=16)
title('Phase of sin(5t)')
grid(True)
show()



#DFT of (1 + 0.1cos(t))cos(10t) with 128 samples
t=linspace(0,2*pi,129);t=t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y=fftshift(fft(y))/128.0
w=linspace(-64,63,129);w=w[:-1]

#Magnitude plot of the DFT of (1 + 0.1cos(t))cos(10t) with 128 samples
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
xlabel(r"$\omega$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right) - 128 samples$")
grid(True)
show()

#Phase plot of the DFT of (1 + 0.1cos(t))cos(10t) with 128 samples
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
title(r"Phase of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
show()

#DFT of (1 + 0.1cos(t))cos(10t) with 512 samples
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]

#Magnitude plot of the DFT of (1 + 0.1cos(t))cos(10t) with 512 samples
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
xlabel(r"$\omega$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)-512 samples$")
grid(True)
show()

#Phase plot of the DFT of (1 + 0.1cos(t))cos(10t) with 512 samples
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
title(r"Phase of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
show()


#Q2: spectrums of (sin(t))^3 and (cos(t))^3

#DFT of (sin(t))^3 with 512 samples
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(sin(t))**3
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]

#Magnitude plot of the DFT of (sin(t))^3
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
xlabel(r"$\omega$",size=16)
title(r"Spectrum of $(sin(t))^3$")
grid(True)
show()

#Phase plot of the DFT of (sin(t))^3
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
title(r"Phase of $(sin(t))^3$")
grid(True)
show()



#DFT of (cos(t))^3 with 512 samples
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=(cos(t))**3
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]

#Magnitude plot of the DFT of (cos(t))^3
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
xlabel(r"$\omega$",size=16)
title(r"Spectrum of $(cos(t))^3$")
grid(True)
show()

#Phase plot of the DFT of (cos(t))^3
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
title(r"Phase of $(cos(t))^3$")
grid(True)
show()


#Q3: spectrum of phase modulated signal cos(20t +5cos(t))

#DFT of cos(20t +5cos(t)) with 512 samples
t=linspace(-4*pi,4*pi,513);t=t[:-1]
y=cos(20*t+5*cos(t))
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]

#Magnitude plot of the DFT of cos(20t +5cos(t))
plot(w,abs(Y),lw=2)
xlim([-40,40])
ylabel(r"$|Y|$",size=16)
xlabel(r"$\omega$",size=16)
title(r"Spectrum of $cos(20t +5cos(t))$")
grid(True)
show()

#Phase plot of the DFT of cos(20t +5cos(t))
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-40,40])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
title(r"Phase of $cos(20t +5cos(t))$")
grid(True)
show()


#Q4: spectrum of Gaussian Function with different sampling rates

#case1 : sampling rate is higher
T=16*pi
N=1024
N2=512

#spectrum of the gaussian function
t = linspace(-T/2,T/2,N+1);t=t[:-1]
y = exp(-(t**2)/2)
Y = fftshift(abs(fft(y)))/N # Finding DFT
Y = Y/(max(Y)*sqrt(2*pi))
w = linspace(-N2*pi/T,N2*pi/T,N+1);w=w[:-1]
fft_gf = (1/sqrt(2*pi))*exp(-(w**2)/2)

#plotting the spectrum
plot(w,abs(Y),lw=2)
xlim(-4,4)
title('Computed fft of gaussian 1')
ylabel(r"$|Y|$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()


#case2 : sampling rate is lower
T=16*pi
N=512
N2=512

#spectrum of the gaussian function along with the ctft of the ACTUAL gaussian function
t = linspace(-T/2,T/2,N+1);t=t[:-1]
y = exp(-(t**2)/2)
Y = fftshift(abs(fft(y)))/N # Finding DFT
Y = Y/(max(Y)*sqrt(2*pi))
w = linspace(-N2*pi/T,N2*pi/T,N+1);w=w[:-1]
fft_gf = (1/sqrt(2*pi))*exp(-(w**2)/2)
error = max(abs(abs(Y)-fft_gf)) #finds the maximum error between the calculated and actual functions

#plotting the spectrum
plot(w,abs(Y),lw=2)
xlim(-4,4)
title('Computed fft of gaussian 2')
ylabel(r"$|Y|$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

#plotting the spectrum of the ACTUAL gaussian function
plot(w,abs(fft_gf),lw=2)
xlim(-4,4)
title('Actual ctft of gaussian')
ylabel(r"$|Y|$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

print('The Maximum error between the computed and actual fft of the gaussian function is %e:' %error)