# -*- coding: utf-8 -*-
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

"""
    bestIter      : My Family Best Perfomanance over time
    bestIndIter   : My Personal Best Performance over time
    mybestIter    : Family with Best Perfomance over time
    bestCommIter  : Community Best Perfomance over time
""" 
import pandas as pd

def initialise_position(individuals,index):
    
    I_update  = pd.DataFrame(columns=individuals, index = index)
    delta_update  = pd.DataFrame(columns=individuals, index = index)
    return I_update,delta_update

def myLocalBest(positionlist,ind_payoffs,ind_position,individualID):
    iteration = pd.DataFrame(columns=['iterations'], index = individualID)
    myLocalBest = pd.DataFrame(index = individualID,columns=positionlist)
    for i in individualID:
        
        best_iteration = ind_payoffs[i].astype(float).idxmax(axis=0)
        myLocalBest_values = ind_position.loc[best_iteration,i].astype(float)
        iteration.loc[i,'iterations'] = best_iteration
        myLocalBest.loc[i,positionlist] = myLocalBest_values
        myLocalBest = myLocalBest.astype(float)
    return myLocalBest,iteration


def myfamGlocalBest(df,fam_payoffs,ind_payoffs,ind_position,myFamID):
    iteration = fam_payoffs[myFamID].astype(float).idxmax(axis=0)
    members = list(df['IndPayoff'][df['Family']==myFamID].index)
    my_bestFamInd = ind_payoffs.loc[iteration,members].astype(float).idxmax(axis=1)
    myfamGlocalBest = ind_position.loc[iteration,my_bestFamInd].astype(float) #global
    
    return myfamGlocalBest,iteration

def famGlobalBest(df,current_iteration,previous_iteration,fam_payoffs,ind_payoffs,ind_position):
    '''
        This means that the Best Family is found in the current Iteration. 
        In the next iteration, you should compare to see if the famGlobalBest value needs to be 
        updated or not.
    '''
    if (previous_iteration == 'iter_-1')|(previous_iteration == 'iter_0'):
        iteration = current_iteration
        bestFamilyID = fam_payoffs.loc[iteration].astype(float).idxmax(axis=1)
        members = list(df['IndPayoff'][df['Family']==bestFamilyID].index)
        bestFambestIndID = ind_payoffs.loc[iteration,members].astype(float).idxmax(axis=1)
        famGlobalBest = ind_position.loc[iteration,bestFambestIndID].astype(float)#global
    else:
        if fam_payoffs.loc[current_iteration].astype(float).max()> fam_payoffs.loc[previous_iteration].astype(float).max():
            iteration = current_iteration
            bestFamilyID = fam_payoffs.loc[iteration].astype(float).idxmax(axis=1)
            members = list(df['IndPayoff'][df['Family']==bestFamilyID].index)
            bestFambestIndID = ind_payoffs.loc[iteration,members].astype(float).idxmax(axis=1)
            famGlobalBest = ind_position.loc[iteration,bestFambestIndID].astype(float)#global
        else:
            iteration = previous_iteration
            bestFamilyID = fam_payoffs.loc[iteration].astype(float).idxmax(axis=1)
            members = list(df['IndPayoff'][df['Family']==bestFamilyID].index)
            bestFambestIndID = ind_payoffs.loc[iteration,members].astype(float).idxmax(axis=1)
            famGlobalBest = ind_position.loc[iteration,bestFambestIndID].astype(float)#global
    
    return famGlobalBest,bestFamilyID,iteration


def commGlobalBest(df,comm_payoffs,fam_payoffs,ind_payoffs,ind_position):
    
    iteration = comm_payoffs['COMMPAYOFF'].astype(float).idxmax(axis=0)
    bestComFamID = fam_payoffs.loc[iteration].astype(float).idxmax(axis=1)
    members = list(df['IndPayoff'][df['Family']==bestComFamID].index)
    bestComFamIndID = ind_payoffs.loc[iteration,members].astype(float).idxmax(axis=1)
    commGlobalBest = ind_position.loc[iteration,bestComFamIndID].astype(float) #global
    
    return commGlobalBest,bestComFamID,iteration
