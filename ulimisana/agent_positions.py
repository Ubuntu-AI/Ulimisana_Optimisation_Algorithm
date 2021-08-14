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

def initialise_position(popSize,dim,time_iter):
    """ This function initialises positions for agents in the community.
    Agents names are defaulted to 'Individual' +str(agent number). eg. 'Individual12', 13 is the agent's identification number. [i.e Prefix = 'Individual']
    Each position variables is given a default name of "IndPosition' + str(dim). eg IndPosition2 indicates the third variable of the dimensions of the objective function dimensions. [i.e Prefix = 'IndPosition]
    Each iteration is given a number and a prefix 'iter_'
    
    
    Parameter:
    ------------
        popSize : The size of the agents population in the community
        dim     : Dimension of the variables of the objective function that these agents should solve for.
        time_iter: The total number of iterations that this agents should take to find the best solution for the objective function.
        
    Return:
    --------
    I_update : This is an empty dataframe whose index is a tuple of the iteration and each of the variables of the dimensions. Its columns are names of agents. 
    delta_update: This an empty dataframe whose index is a tuple of the iteration and each of the variables of the dimensions. Its columns are names of agents.
    """
    individuals = ['Individual'+str(i) for i in range(0,popSize)]
    iter_dx = ['iter_'+str(i) for i in range(0,(time_iter))]
    positionlist = ['IndPosition'+str(i) for i in range(0,dim)]
    index = pd.MultiIndex.from_product([iter_dx, positionlist],
                                       names=['iter', 'position'])
    I_update  = pd.DataFrame(columns=individuals, index = index)
    delta_update  = pd.DataFrame(columns=individuals, index = index)
    return I_update,delta_update

def myLocalBest(positionlist,ind_payoffs,ind_position,individualID):
    """
    This function calculates the agent's local best position
    
    Parameter:
    -----------
        positionlist: list of dimension variable names
        ind_payoffs : Individual Payoffs DataFrame
        ind_position : Individual's positions
        individualID : Individual's Identification number in the community
    
    Return:
    ---------
    myLocalBest : Agent's best local position
    iteration   : The iteration at which the agent was at its best performance. 
    """
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
    """
    This function calculates the agent's family's best  perfoming individual's position
    
    Parameter:
    -----------
        df : This is a dataframe with the information about the agents. Includes their family, position, age and payoff details.
        fam_payoffs: This indicates the familys paoff's dataframe
        ind_payoffs : Individual Payoffs DataFrame
        ind_position : Individual's positions
        myFamID : Individual's Family Identification number in the community
    
    Return:
    ---------
    myfamGlocalBest : Agent's Family's best agent's best position
    iteration   : The iteration at which this agent's family was at its best performance. 
    """
    iteration = fam_payoffs[myFamID].astype(float).idxmax(axis=0)
    members = list(df['IndPayoff'][df['Family']==myFamID].index)
    my_bestFamInd = ind_payoffs.loc[iteration,members].astype(float).idxmax(axis=1)
    myfamGlocalBest = ind_position.loc[iteration,my_bestFamInd].astype(float) #global
    
    return myfamGlocalBest,iteration

def famGlobalBest(df,current_iteration,previous_iteration,fam_payoffs,ind_payoffs,ind_position):
    """
        This means that the Best Family is found in the current Iteration. 
        In the next iteration, you should compare to see if the famGlobalBest value needs to be 
        updated or not.
        
    
    This function calculates the best famil's best agent's position
    
    Parameter:
    -----------
        df : This is a dataframe with the information about the agents. Includes their family, position, age and payoff details.
        current_iteration : This indicates the current iteration that the whole solution search is in.
        previous_iteration : This indiacate teh previous iteration that the whole solution search was in. 
        fam_payoffs: This indicates the familys paoff's dataframe
        ind_payoffs : Individual Payoffs DataFrame
        ind_position : Individual's positions
    
    Return:
    ---------
    famGlobalBest : Best Global Family's best agent's best position
    bestFamilyID : ID of the Best Global Family 
    iteration   : The iteration at which this agent's family was at its best performance. 
    """
  
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
    """
    This function calculates the best agents from the best perfoming family when the community was at best performance
    
    Parameter:
    -----------
        df : This is a dataframe with the information about the agents. Includes their family, position, age and payoff details.
        comm_payoffs : This a community payoff's dataframe
        fam_payoffs: This indicates the familys paoff's dataframe
        ind_payoffs : Individual Payoffs DataFrame
        ind_position : Individual's positions
    
    Return:
    ---------
    commGlobalBest : Best Global Family's best agent's best position when the community was at best performance 
    bestComFamID : ID of the Best Global Family in the communitie's global performance.
    iteration   : The iteration at which this agent's family was at its best performance. 
    """
    iteration = comm_payoffs['COMMPAYOFF'].astype(float).idxmax(axis=0)
    bestComFamID = fam_payoffs.loc[iteration].astype(float).idxmax(axis=1)
    members = list(df['IndPayoff'][df['Family']==bestComFamID].index)
    bestComFamIndID = ind_payoffs.loc[iteration,members].astype(float).idxmax(axis=1)
    commGlobalBest = ind_position.loc[iteration,bestComFamIndID].astype(float) #global
    
    return commGlobalBest,bestComFamID,iteration
