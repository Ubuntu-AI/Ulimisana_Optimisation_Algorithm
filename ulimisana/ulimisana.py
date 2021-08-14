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

### INPUTS ###
time_iter        : Number of iterations
lb : Lower Boundary
ub : Upper Boundary
dim : Dimensions of the objective function

popSize          :The prefered population size
NoOfFamilies     : The prefered number of families
fam_aveThreshold : The average threshold to show family is well off [0,1]
com_aveThreshold : The average threshold to show community is well off [0,1]
ageAverage       : Average age of individuals in the each family
ageStdev         : Standard Deviation of age per family
phi              : [0,1] Used to specify how much the providers and wealthy families 
                       should contibute to family and community respectively


Trustworthiness
epsilon        : 0.15 
r              : 0.7
trustThreshold : Trust Threshold for a family to become an advisor [0,1]


objFunction   : wayburnSeader2Function

Transformation Functions:
sigFun_term   : [no_tran_term, tanh, logistic, arctan, gudermanannian, algebraic, erf]
sigFun_weight : [no_tran_weight, tanh, logistic, arctan, gudermanannian, algebraic, erf]

### OUTPUTS ###

x_info : Info about individual characteristics (Age and Family ID)
x_pos  : Individual Position
ind_val :Individuals Payoff/ Objective Value
fam_val :Family's Payoff (Ubuntu Incentive)
comm_val : Community Payoff (Ubuntu Incentive)
adv_trust : Trust towards advisors
trust     : Trust towards all other families.
"""

import agent_positions as ps
import trustworthinessFunction as twf
import ubuntuIncentives as ui
import numpy as np
import pandas as pd

def ulimisana(time_iter,popSize,NoOfFamilies,ageAverage,ageStdev,
              objFunction,dim,lb,ub,sigFun_term,sigFun_weight,fam_aveThreshold =0.3,com_aveThreshold=0.3,phi=0.7,epsilon=0.15,r=0.7,
              trustThreshold=0.65):
     
    """
    ### INPUTS ###
    time_iter        : Number of iterations
    popSize          :The prefered population size
    NoOfFamilies     : The prefered number of families
    
    ageAverage       : Average age of individuals in the each family
    ageStdev         : Standard Deviation of age per family
    objFunction      : The function to be solved
    dim              : Dimensions of the objective function
    lb               : Lower Boundary
    ub               : Upper Boundary
    sigFun_term      : [no_tran_term, tanh, logistic, arctan, gudermanannian, algebraic, erf]
    sigFun_weight    : [no_tran_weight, tanh, logistic, arctan, gudermanannian, algebraic, erf]
    fam_aveThreshold : The average threshold to show family is well off [0,1]
    com_aveThreshold : The average threshold to show community is well off [0,1]
    phi              : [0,1] Used to specify how much the providers and wealthy families 
                           should contibute to family and community respectively

    ## Trustworthiness
    epsilon         : 0.15 
    r               : 0.7
    trustThreshold  : Trust Threshold for a family to become an advisor [0,1]
    
    
    ## Transformation Functions:
    

    ### OUTPUTS ###

    x_info      : Info about individual characteristics (Age and Family ID)
    x_pos       : Individual Position
    ind_val     :Individuals Payoff/ Objective Value
    fam_val     :Family's Payoff (Ubuntu Incentive)
    comm_val    : Community Payoff (Ubuntu Incentive)
    adv_trust   : Trust towards advisors
    trust       : Trust towards all other families.
    """
    
    individuals = ['Individual'+str(i) for i in range(0,popSize)]
    famID = ['Family'+str(i) for i in range(0,NoOfFamilies)]
    df,ind_position,ind_payoffs,fam_payoffs,comm_payoffs = ui.initialise_ubuntuIncentives(
            NoOfFamilies,popSize,ageAverage,ageStdev,dim,lb,ub,time_iter)
    iter_ratings,iter_evaluatedTrustworthiness,iter_adv_Trustworthiness,iter_trustNetwork = twf.initialise_iter_trust(NoOfFamilies)
    trustNetwork,adv_trustworthiness = twf.initialise_trust(NoOfFamilies)
    I_update,delta_update = ps.initialise_position(popSize,dim,time_iter)
    
    evaluatedTrustworthiness = pd.DataFrame(columns = famID, index=famID)
    evaluatedTrustworthiness = evaluatedTrustworthiness.fillna(0.5)
    
    positionlist = list(df.columns)
    positionlist.remove('age')
    positionlist.remove('Family')
    if 'IndPayoff' in positionlist:
        positionlist.remove('IndPayoff')

    for time_iter_idx in range(0,time_iter):  
        iter_idx = 'iter_'+str(time_iter_idx)
        pre_iter_idx = 'iter_'+str(time_iter_idx-1)
        current_iteration = iter_idx
        previous_iteration = pre_iter_idx
        print('Iteration: ',time_iter_idx)
        
        ind_position, ind_payoffs, fam_payoffs, comm_payoffs = ui.ubuntuIncentives(objFunction,df,
                          ind_position, ind_payoffs, fam_payoffs, comm_payoffs
                         ,time_iter_idx,fam_aveThreshold,com_aveThreshold ,phi)
        if time_iter_idx%52 ==0: 
            df['age'] += 1
    
        #''' Trustworthiness'''
        evaluatedTrustworthiness,adv_trustworthiness,trustNetwork,ratings = twf.trustworthinessFunction(
                            trustNetwork,adv_trustworthiness,trustThreshold,epsilon,r)
        evaluatedTrustworthiness['Time'] = iter_idx
        trustNetwork['Time'] = iter_idx
        adv_trustworthiness['Time'] = iter_idx
        ratings['Time'] = iter_idx
    
        iter_adv_Trustworthiness = iter_adv_Trustworthiness.append(adv_trustworthiness)
        iter_evaluatedTrustworthiness = iter_evaluatedTrustworthiness.append(evaluatedTrustworthiness)
        iter_trustNetwork = iter_trustNetwork.append(trustNetwork)
        iter_ratings = iter_ratings.append(ratings)
        ratings =ratings.drop(columns='Time')
        trustNetwork =trustNetwork.drop(columns='Time')
        evaluatedTrustworthiness =evaluatedTrustworthiness.drop(columns='Time')
        adv_trustworthiness = adv_trustworthiness.drop(columns='Time')
        #''' Update Individual Positions'''
    
        for i in famID:
            globalbestComm,bestComFamID,bestComm   = ps.commGlobalBest(df,comm_payoffs,fam_payoffs,ind_payoffs,ind_position) 
            trustfamBest_comm = iter_evaluatedTrustworthiness[iter_evaluatedTrustworthiness['Time'] == bestComm]
            trustfamCurrent = iter_evaluatedTrustworthiness[iter_evaluatedTrustworthiness['Time'] == iter_idx]
            currentFam_commTrust = np.mean(trustfamCurrent[i])
            bestFam_commTrust_comm = np.mean(trustfamBest_comm[i])
            weight_5 = (np.exp(currentFam_commTrust) - np.exp(bestFam_commTrust_comm))/(np.exp(currentFam_commTrust) + np.exp(bestFam_commTrust_comm))
            
            famGlobalBest,mybestFamily,famGlobalBest_best_iteration    = ps.famGlobalBest(
                    df,current_iteration,previous_iteration,fam_payoffs,ind_payoffs,ind_position)
            advisorsNet = trustNetwork.loc[i] #Extract advisors from Trust Network
            advisors = advisorsNet.index[advisorsNet].tolist()
            weight_1_a = np.exp(adv_trustworthiness.loc[i,mybestFamily])
            if advisors ==[]:
                weight_1_b =0
            else:
               weight_1_b = np.exp(np.mean(adv_trustworthiness.loc[advisors,mybestFamily]))
            weight_1 = (weight_1_a  - weight_1_b)/(weight_1_a  + weight_1_b)
            
            glocalbestFamily,bestIter = ps.myfamGlocalBest(df,fam_payoffs,ind_payoffs,ind_position,i)
            trustfamBest = iter_evaluatedTrustworthiness[iter_evaluatedTrustworthiness['Time'] == bestIter]
            trustfamCurrent = iter_evaluatedTrustworthiness[iter_evaluatedTrustworthiness['Time'] == iter_idx]
            bestFam_commTrust = np.mean(trustfamBest[i])
            currentFam_commTrust = np.mean(trustfamCurrent[i])
            weight_2 = (np.exp(currentFam_commTrust) - np.exp(bestFam_commTrust))/(np.exp(currentFam_commTrust) + np.exp(bestFam_commTrust))
    
            for j in df['IndPayoff'][df['Family']==i].index.tolist():
               
               bestIndIter = ind_payoffs[j].astype(float).idxmax(axis=0)
               bestIndposition = ind_position.loc[bestIndIter,j]
               best_famNetMembership = sum(iter_trustNetwork[iter_trustNetwork['Time'] == bestIndIter][i])
               current_famNetMembership = sum(iter_trustNetwork[iter_trustNetwork['Time'] == iter_idx][i])
               if current_famNetMembership >0:
                   weight_3 = (np.exp(current_famNetMembership) - np.exp(best_famNetMembership))/(np.exp(current_famNetMembership) + np.exp(best_famNetMembership))
               else:
                   weight_3 = 0
            
               term_1 = (famGlobalBest    - df.loc[j,positionlist]).astype(float)
               term_2 = (glocalbestFamily - df.loc[j,positionlist]).astype(float)
               term_3 = (bestIndposition  - df.loc[j,positionlist]).astype(float)
               term_5 = (globalbestComm   - df.loc[j,positionlist]).astype(float)
               
               delta_position_update = sigFun_term(term_1)*sigFun_weight(weight_1) + sigFun_term(
                       term_2)*sigFun_weight(weight_2) + sigFun_term(term_3)*sigFun_weight(
                               weight_3) + sigFun_term(term_5)*sigFun_weight(weight_5)
               
               delta_update.loc[iter_idx,j] = np.array(delta_position_update)
               ind_pos_update = df.loc[j,positionlist] + delta_position_update
               ind_pos_update = np.clip(ind_pos_update,lb,ub)
               df.loc[j,positionlist] = np.array(ind_pos_update)
               I_update.loc[iter_idx,j] = np.array(ind_pos_update)
    
        IndividualPosition = df[positionlist]
        x = IndividualPosition
        IndividualPayoff =  -1*objFunction(x)
        ind_payoffs.loc[iter_idx,individuals] =  np.transpose(IndividualPayoff)
        df['IndPayoff'] = np.array(IndividualPayoff)
        print('mean: ',np.mean(df['IndPayoff']))
        print('mim : ',np.min(df['IndPayoff']))
        print('max : ',np.max(df['IndPayoff']))

    return df,I_update,ind_payoffs,fam_payoffs,comm_payoffs,iter_adv_Trustworthiness,iter_evaluatedTrustworthiness