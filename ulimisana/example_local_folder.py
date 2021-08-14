# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 13:25:31 2021

@author: Ngwaniwapho
"""

import testFunctions as tf
import transformFunctions as trf
import ulimisana as uoa
# Total Number of Iterations
time_iter = 300
# Lower bound, Upper bound, Test Function Dimension
lb = -11
ub =  11
dim = 2

# Community size
popSize = 100
# Number of families in community
NoOfFamilies = 5
#Average Weight family and community Payoffs (Fitness values)
fam_aveThreshold = 0.3
com_aveThreshold = 0.3
# Age distributions to determine how many dependents and providers in each family
ageAverage = 35
ageStdev = 7
# Use in determining the Ubuntu Incentive Scheme
phi = 0.7

'''Trustworthiness'''
epsilon = 0.15 
r = 0.7
trustThreshold =0.45
errorDec= 10**(-3)

objFunction   = tf.penHolderFunction # Make sure the dimensions, lb and ub chosen align with test function.
sigFun_term   = trf.tanh # Choose between linear and sigmoid transformation function
sigFun_weight = trf.tanh #Choose between linear and sigmoid transformation function

'''
 x_info    : gives you information about each agent. Their age, position at the end of iterations, 
                     their objective value (payoff) and family they belong to.
 x_pos     : gives you the changes in position over all the iterations
 ind_val   : gives you the individual's payoffs for each iteration
 fam_val   : gives you the family' payoffs for each iteration
 com_val   : gives you the cimmunity's pauoffs for each iteration
 adv_trust : gives you the changes of trustworthiness each family had towards their advisrors. 
 trust     : gives you the changes of trustworthiness each family had towards all families (computed using advisor's trustworthiness)

'''

x_info,x_pos,ind_val,fam_val,comm_val,adv_trust,trust= uoa.ulimisana(time_iter,popSize,NoOfFamilies,ageAverage,ageStdev,
              objFunction,dim,lb,ub,sigFun_term,sigFun_weight,fam_aveThreshold,com_aveThreshold,phi,epsilon,r,
              trustThreshold)