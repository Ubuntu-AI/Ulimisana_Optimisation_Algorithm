"""
Created on Sat Jul 20 18:42:31 2019

@authors: Tshifhiwa Maumela, Fulufhelo Ṋelwamondo, Tshilidzi Marwala

Correspondance: josh.maumela@gmail.com



https://ieeexplore.ieee.org/document/9205897
@ARTICLE{9205897,  author={T. {Maumela} and F. {Ṋelwamondo} and T. {Marwala}},  
journal={IEEE Access},   
title={Introducing Ulimisana Optimisation Algorithm based on Ubuntu Philosophy},   
year={2020},  
volume={},  
number={},  
pages={1-1},}

"""

import numpy as np
from scipy import special

def tanh(x):
#    x = x.astype(float)
    return np.tanh(x)

def no_tran_weight(x):
    return x

def linear(x):
    return x
    
def no_tran_term(x):
    return x

def logistic(x):
    logistic = 1/(1+np.exp(-x))
    return logistic

def arctan(x):
#    x = x.astype(float)
    return np.arctan(x)

def gudermanannian(x):
#    x = x.astype(float)
    gudermanannian = 2*np.arctan(np.tanh(x/2))
    return gudermanannian

def algebraic(x):
#    x = x.astype(float)
    algebraic = x/(np.sqrt(1+x**2))
    return algebraic

def erf(x):
#    x = x.astype(float)
    return special.erf(x)

