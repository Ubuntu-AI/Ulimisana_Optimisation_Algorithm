# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:50:43 2020

@author: Ngwaniwapho
"""
import numpy as np

__all_ = ['easomFunction','wayburnSeader2Function']
def easomFunction(x):
#    −100 ≤ xi ≤ 100. The global minimum is located at x∗ = f(π, π),f(x∗) = −1.
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    easomfunction = -np.cos(x1)*np.cos(x2)*np.exp(-1*(x1-np.pi)**2 -(x2 - np.pi)**2)
    return easomfunction

def bealeFunction(x):
#    subject to −4.5 ≤ xi ≤ 4.5. The global minimum is located at x∗ = (3, 0.5), f(x∗) = 0.
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    bealefunction = (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2
    return bealefunction

def matyasFunction(x):
    #subject to −10 ≤ xi ≤ 10. The global minimum is located at x∗ = f(0, 0), f(x∗) = 0.
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    matayasfunction = 0.26*(x1**2 + x2**2)- 0.48*x1*x2
    return matayasfunction

def bohachevsky1Function(x):
    #subject to −100 ≤ xi ≤ 100. The global minimum is located at x∗ = f(0, 0), f(x∗) = 0.
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    bohachevsky1function = x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1)-0.4*np.cos(4*np.pi*x2)+0.7
    return bohachevsky1function

def penHolderFunction(x):
    #subject to −11 ≤ xi ≤ 11. The four global minima are located at x∗ = f(±9.646168, ±9.646168), f(x∗) = −0.96354.
#    f(i)=-exp(-1/abs(cos(x1(j))*cos(x2(i))*exp(abs(1-sqrt(x1(j).^2+x2(i).^2)/pi))))
#    penHolderfunction = -np.exp(np.abs(np.cos(x1)*np.cos(x2)*np.exp(np.abs(1-(np.sqrt(x1**2 + x2**2))/np.pi)))**(-1))
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    penHolderfunction = -np.exp(-1/np.abs(np.cos(x1)*np.cos(x2)*np.exp(np.abs(1-(np.sqrt(x1**2 + x2**2))/np.pi))))
    return penHolderfunction

def wayburnSeader2Function(x):
#    subject to −500 ≤ 500. The global minimum is located at x∗ = f{(0.2, 1), (0.425, 1)}, f(x∗) = 0.
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    wayburn = (1.613 - 4*(x1 - 0.3125)**2 - 4*(x2 - 1.625)**2)**2 + (x2 - 1)**2
    return wayburn

def schaffer1Function(x):
#    −100 ≤ xi ≤ 100. The global minimum is located at x∗ = f(0, 0),f(x∗) = 0.
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    numeratorcomp   = (np.sin((x1**2 + x2**2)**2)**2) - 0.5
    denominatorcomp = (1 + 0.001 * (x1**2 + x2**2))**2 
    scahffer1function = 0.5 + numeratorcomp /denominatorcomp
    return scahffer1function

def wolfeFunction(x):
#    0 ≤ xi ≤ 2. The global minimum is located at x∗ = f(0,0, 0),f(x∗) = 0.
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    x3 = x['IndPosition2']
    wolfefunction = (4/3)*(((x1**2 + x2**2) - (x1*x2))**(0.75)) + x3
    return wolfefunction

def ackley2Function(x):
#    subject to −32 ≤ xi ≤ 32. The global minimum is located at origin x∗ = (0, 0),f(x∗) = −200.
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    ackley2function = -200*np.exp((-0.2*np.sqrt(x1**2 +x2**2)))
    return ackley2function

def goldsteinpriceFunction(x):
    #subject to −2 ≤ xi ≤ 2. The global minimum is located at x∗ = f(0,−1), f(x∗) = 3.
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    term1 = (x1 + x2 + 1)**2
    term3 = (2*x1 - 3*x2)**2
    goldsteinpricefunction = (1 + term1*(19 - 14*x1 +3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))*(30 + term3*(18 - 32*x1 + 12*x1**2+ 48*x2 - 36*x1*x2 + 27*x2**2))
    return goldsteinpricefunction

def boothFunction(x):
#subject to −10 ≤ xi ≤ 10. The global minimum is located at x∗ = f(1,3), f(x∗) = 0.
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    boothfunction =((x1+2*x2-7)**2+(2*x1+x2-5)**2)
    return boothfunction

def brentFunction(x):
#subject to −20 ≤ xi ≤ 0. The global minimum is located at x∗ = f(-10,-10), f(x∗) = np.exp(-200)
    x1 = x['IndPosition0']
    x2 = x['IndPosition1']
    brentfunction = (x1 + 10)**2 + (x2 + 10)**2 + np.exp(-x1**2 - x2**2)
    return brentfunction

def powellsumFunction(x):
#subject to −1 ≤ xi ≤ 1. The global minimum is located at x∗ = f(0,...,0), f(x∗) = 0
    n = (x.shape)[1]
    absx = np.abs(x)

    powellsumfunction = 0
    for i in range(n):
        powellsumfunction = powellsumfunction + (absx.iloc[:, i]**((i+1) + 1))
    return powellsumfunction 