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
from numpy.random import randint
from numpy.random import randn
from numpy.random import uniform
import numpy as np
import pandas as pd

def initialise_ubuntuIncentives(NoOfFamilies,popSize,ageAverage,ageStdev,dim,lb,ub,time_iter):
    """
    This function creates empty dataframes that are used in the Ubuntu Incentive Mechanism function
    
    Parameter:
    -----------
        NoOfFamilies     : The prefered number of families
        popSize          :The prefered population size
        ageAverage       : Average age of individuals in the each family
        ageStdev         : Standard Deviation of age per family
        dim              : Dimensions of the objective function
        lb               : Lower Boundary
        ub               : Upper Boundary
        time_iter        : total number of iterations.
    
    Return:
    ---------
        df              : This is an empty dataframe which will contain all information about the agent including their Family, age, position and payoffs
        ind_position    : This is an empty dataframe which will contain the updated agent's position over time
        ind_payoffs     : This is an empty dataframe which will contain the updated agent's payoffs over time
        fam_payoffs     : This is an empty dataframe which will contain the updated agent's family's payoffs over time
        comm_payoffs    : This is an empty dataframe which will contain the updated agent's community's payoffs over time
    """


    individuals = ['Individual'+str(i) for i in range(0,popSize)]
    positionlist = ['IndPosition'+str(i) for i in range(0,dim)]
    Family = ['Family'+str(i) for i in randint(0,NoOfFamilies,popSize)]
    lists =['Family'] +positionlist
    df = pd.DataFrame(columns = lists, index=individuals)
    df['Family']= Family
    famID = df['Family'].unique()
    values = randn(popSize)
    scaled_values = ageAverage + values*ageStdev
    scaled_values = np.abs(np.round(scaled_values,0))
    df['age'] = scaled_values
    IndividualPosition =uniform(lb,ub,(popSize,dim)) 
    df[positionlist] = IndividualPosition
    
    
    iter_dx = ['iter_'+str(i) for i in range(0,(time_iter))]
    fam_payoffs = pd.DataFrame(columns=famID, index = iter_dx)
    ind_payoffs = pd.DataFrame(columns=individuals, index = iter_dx)
    index = pd.MultiIndex.from_product([iter_dx, positionlist],
                                       names=['iter', 'position'])
    ind_position = pd.DataFrame(columns=individuals, index = index)
    comm_payoffs = pd.DataFrame(columns= ['COMMPAYOFF'], index = iter_dx)
    
    return df,ind_position,ind_payoffs,fam_payoffs,comm_payoffs
    
def ubuntuIncentives(objFunction,df,ind_position,ind_payoffs,fam_payoffs, 
                     comm_payoffs,time_iter,fam_aveThreshold = 0.3,com_aveThreshold = 0.3,phi = 0.7):
    """
    
    Parameters:
    ----------------
            objFunction             : This is the objective function whose minimum values we want to determine. 
            df                      : This is a dataframe which contains all information about the agent including their Family, age, position and payoffs
            famMinPayoffThreshold   :   This is s scaler that ranges [0,1] which determines the factor by which the poor family's payoffs should be lower than. 
                                        This is multiplied to the average payfoofs of individuals in the family.
            commMinPayoffThreshold  : This is s scaler that ranges [0,1] which determines the factor by which the poor family's payoffs should be lower than. 
                                        This is multiplied to the average payfoofs of individuals in the community
            phi + rho = 1           : These are used as weighted average where the direction of each is determined by the whether the minimum family or community payoff was met.
    
    Return:
    --------------
                ind_position    : This is an empty dataframe which will contain the updated agent's position over time
                ind_payoffs     : This is an empty dataframe which will contain the updated agent's payoffs over time
                fam_payoffs     : This is an empty dataframe which will contain the updated agent's family's payoffs over time
                comm_payoffs    : This is an empty dataframe which will contain the updated agent's community's payoffs over time
                
    """
    famID = df['Family'].unique()
    df2 = pd.DataFrame(columns = ['Members','Dependents',
                                  'Providers','familyPayoff'], index= famID)
                                  
    positionlist = list(df.columns)
    positionlist.remove('age')
    positionlist.remove('Family')
    if 'IndPayoff' in positionlist:
        positionlist.remove('IndPayoff')
    pre_iter_idx = 'iter_'+str(time_iter-1)
    IndividualPosition = df[positionlist]

    x = IndividualPosition
    iter_idx = 'iter_'+str(time_iter)
    IndividualPayoff =  -1*objFunction(x)
    df['IndPayoff'] = IndividualPayoff
    min_FamilyPayoff = fam_aveThreshold*np.mean(IndividualPayoff)
    for i in famID:
        fam = df[df['Family']==i]
        df2['Members'][i] = len(fam)
        a = fam['age'] <20
        b = fam['age'] >60      
        c = a|b
        dependentsPop = fam[c]
        providersPop = fam[~c]
        dependentsPayoff = sum(dependentsPop['IndPayoff'])
        providersPayoff = sum(providersPop['IndPayoff'])
        df2['Dependents'][i] = len(dependentsPop)
        df2['Providers'][i]= len(providersPop)
            
        if dependentsPayoff > min_FamilyPayoff:
            rho = phi
        elif dependentsPayoff <= min_FamilyPayoff:
            rho = 1 - phi
        familyPayoff = rho*providersPayoff + (1 - rho)*dependentsPayoff
        df2['familyPayoff'][i] = familyPayoff
        
    min_communityPayoff = com_aveThreshold*np.mean(df2['familyPayoff'])
    a = df2['familyPayoff'] > min_communityPayoff
    wealthyFamilyPop = df2[a]
    poorFamilyPop = df2[~a]
    wealthyPayoff = sum(wealthyFamilyPop['familyPayoff'])
    poorPayoff = sum(poorFamilyPop['familyPayoff'])
    
    if np.mean(poorPayoff) > min_communityPayoff:
        rho = phi
    elif np.mean(poorPayoff) <= min_communityPayoff:
        rho = 1 - phi
    communityPayoff = rho*wealthyPayoff + (1-rho)*poorPayoff
    comm_payoffs.loc[iter_idx] = communityPayoff    

    if time_iter ==0:
        commPayoffDelta = 0
    elif comm_payoffs.loc[pre_iter_idx,'COMMPAYOFF']==0:
        commPayoffDelta = (comm_payoffs.loc[iter_idx]['COMMPAYOFF']  
                          -comm_payoffs.loc[pre_iter_idx]
                          ['COMMPAYOFF']) /(1)
    else:
       commPayoffDelta = (comm_payoffs.loc[iter_idx]['COMMPAYOFF']  
                          -comm_payoffs.loc[pre_iter_idx]
                          ['COMMPAYOFF']) /comm_payoffs.loc[pre_iter_idx,'COMMPAYOFF'] # communityPayoff(t+1) - communityPayoff(t)
       
    for i in famID:
        familyPayoff = df2['familyPayoff'][i]
        familyPayoff += (commPayoffDelta/len(famID))
        df2['familyPayoff'][i] = familyPayoff
        fam_payoffs.loc[iter_idx,i] = familyPayoff
        
        if time_iter ==0:
            famPayoffDelta = 0
        elif  fam_payoffs.loc[pre_iter_idx,i]==0:
            famPayoffDelta = (fam_payoffs.loc[iter_idx,i] 
            - fam_payoffs.loc[pre_iter_idx,i])/(1)
        else:
            famPayoffDelta = (fam_payoffs.loc[iter_idx,i] 
            - fam_payoffs.loc[pre_iter_idx,i])/fam_payoffs.loc[pre_iter_idx,i] # (t+1) - (t)
    #''' Individual Payoff update '''
        indPayoffUpdate = df[df['Family']==i]['IndPayoff']  + (famPayoffDelta/len(df[df['Family']==i]))
        df['IndPayoff'][df['Family']==i] = indPayoffUpdate
        
        c = list(df['IndPayoff'][df['Family']==i].index)
        ind_payoffs.loc[iter_idx,c] =  np.transpose(indPayoffUpdate)
        ind_position.loc[iter_idx,c] = np.array(np.transpose(df.loc[c,positionlist]))
    
    return ind_position, ind_payoffs, fam_payoffs, comm_payoffs