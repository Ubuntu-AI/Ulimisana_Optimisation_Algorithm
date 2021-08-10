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
import numpy as np
import pandas as pd


def initialise_trust(NoOfFamilies,famID):
#    seed(1)
    trustNetwork = np.random.randint(2, size=(NoOfFamilies,NoOfFamilies))
    trustNetwork = pd.DataFrame(trustNetwork,    # values
              index=famID,    # 1st column as index
              columns=famID)  # 1st row as the column names
    
    trustworthMatrix= np.random.randint(1, size=(NoOfFamilies,NoOfFamilies))
    trustworthMatrix = pd.DataFrame(trustworthMatrix,    # values
              index=famID,    # 1st column as index
              columns=famID)  # 1st row as the column names
    
    trustNetwork = trustNetwork==1
    
    iter_evaluatedTrustworthiness = pd.DataFrame(columns = famID)
    iter_evaluatedTrustworthiness['Time'] = []
    iter_adv_Trustworthiness = pd.DataFrame(columns = famID)
    iter_adv_Trustworthiness['Time'] = []
    iter_trustNetwork = pd.DataFrame(columns = famID)
    iter_trustNetwork['Time'] = []
    iter_ratings = pd.DataFrame(columns = famID)
    iter_ratings['Time'] = []
    return trustNetwork,trustworthMatrix,iter_ratings,iter_evaluatedTrustworthiness,iter_adv_Trustworthiness,iter_trustNetwork


def trustworthinessFunction(df2,trustNetwork,adv_trustworthiness,famID,trustThreshold,epsilon,r):
    

    N_min = -1/(2*epsilon**2) * np.log((1-r)/2)
    evaluating_Family = df2
    
    NoOfFamilies = len(famID)

    ratings = np.random.randint(2, size=(NoOfFamilies,NoOfFamilies))
    np.fill_diagonal(ratings,0)
    ratings = pd.DataFrame(ratings,    # values
              index=famID,    # 1st column as index
              columns=famID)  # 1st row as the column names

    D_positive = pd.DataFrame(columns = famID, index=famID)
    D_positive = D_positive.fillna(0)
    D_negative = pd.DataFrame(columns = famID, index=famID)
    D_negative = D_negative.fillna(0)
    
    evaluatedTrustworthiness = pd.DataFrame(columns = famID, index=famID)
    evaluatedTrustworthiness = D_negative.fillna(0.5)
    
    for i in evaluating_Family.index.tolist():
        advisorsNet = trustNetwork.loc[i] #Extract advisors from Trust Network
        advisors = advisorsNet.index[advisorsNet].tolist()
        evaluatedFamilies = evaluating_Family.drop(index=i)
#        print(i)
        if i in advisors:
            advisors.remove(i)
    #        print(i)
        
        for j in advisors:
            #print(j)
            
            advisor_Ratings = ratings.loc[j]#[dependentFamilies.index.tolist()]
            evaluating_Family_Ratings = ratings.loc[i]#[dependentFamilies.index.tolist()]
            
            evaluating_FamilyRatings = evaluating_Family_Ratings
            advisorRatings = advisor_Ratings
            advisorRatings = advisorRatings.drop(index=j)
            advisorRatings = advisorRatings.drop(index=i)
            evaluating_FamilyRatings =evaluating_FamilyRatings.drop(index=j)
            evaluating_FamilyRatings =evaluating_FamilyRatings.drop(index=i)
            
            if j in evaluatedFamilies.index.tolist():
                ### Have to change this so that you don't loose the dependent family members with each iteration 
                evaluated_Families = evaluatedFamilies
                evaluated_Families = evaluated_Families.drop(index=j)
            else:
                evaluated_Families = evaluatedFamilies
            
            """Private Reputation"""
            N_all = len(evaluating_FamilyRatings)
            N_f = sum((evaluating_FamilyRatings +
                      advisorRatings)==2)
            alpha = N_f + 1
            beta = N_all - N_f + 1
            R_pri = alpha / (alpha + beta)
            
            if N_all < N_min:
                w = N_all/N_min
            else:
                w = 1
            
            """Public Reputation"""
            N_all = len(advisorRatings)
            N_f = sum(advisorRatings == 1)
            alpha = N_f + 1
            beta = N_all - N_f + 1
            R_pub = alpha / (alpha + beta)
            
            Trustworthiness = w*R_pri + (1-w)*R_pub
            adv_trustworthiness[j].loc[i] = Trustworthiness
            
            
            for d in evaluated_Families.index.tolist():
                
                if advisorRatings.loc[d] == 1:
                    N_pos = 1
                    N_neg = 0
                else:
                    N_pos = 0
                    N_neg = 1
                    
                D_positive[j].loc[d] = 2*Trustworthiness*N_pos/((
                        1-Trustworthiness)*(N_pos+N_neg)+2)
                D_negative[j].loc[d] = 2*Trustworthiness*N_neg/((
                        1-Trustworthiness)*(N_pos+N_neg)+2)  
                publicRating = (sum(D_positive[j])+1)/(sum(D_positive[j]) +
                                sum(D_negative[j]) +2)
                
                if evaluating_FamilyRatings.loc[d] == 1:
                    N_pos = 1
                    N_neg = 0
                else:
                    N_pos = 0
                    N_neg = 1
                #providerFamilyRatings.loc[dependentFamilies.index.tolist()]
                privateRating = (N_pos +1)/(N_pos + N_neg +2)
                
                N_all = len(evaluated_Families)
                if N_all < N_min:
                    w = N_all/N_min
                else:
                    w = 1
                Trustworthiness = w*privateRating + (1-w)*publicRating
                evaluatedTrustworthiness[d].loc[i] = Trustworthiness
    trustNetwork =+ evaluatedTrustworthiness >= trustThreshold
    
    return evaluatedTrustworthiness,adv_trustworthiness,trustNetwork,ratings
