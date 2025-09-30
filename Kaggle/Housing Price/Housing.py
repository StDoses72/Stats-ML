import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def main():
    #import file
    housingTrainDataFile = r'D:\Kaggle\Housing\train.csv'
    housingTestDataFile = r'D:\Kaggle\Housing\test.csv'
    trainData = pd.read_csv(housingTrainDataFile)
    testData = pd.read_csv(housingTestDataFile)
    
    #Chaning Zoning Area from word into value
    MSZoningMap = {'A':0,'C (all)':1,'FV':3,'I':4,'RH':5,'RL':6,'RP':7,'RM':8}
    trainData['MSZoningNum']=trainData['MSZoning'].map(MSZoningMap)
    testData['MSZoningNum']=testData['MSZoning'].map(MSZoningMap)

    #Filling LotFrontage Missing part
    trainData['LotFrontage'] = trainData['LotFrontage'].fillna(trainData['LotFrontage'].median)
    testData['LotFrontage'] = testData['LotFrontage'].fillna(trainData['LotFrontage'].median)

    #Chaning Street paved or not
    pavedMap = {'Pave':1,'Grvl':0}
    trainData['StreetPaved'] = trainData['Street'].map(pavedMap)
    testData['StreetPaved'] = testData['Street'].map(pavedMap)

    #Shape of property
    trainData['isRegularShape'] = 0
    testData['isRegularShape'] = 0
    trainData.loc[trainData['LotShape']=='Reg','isRegularShape'] = 1
    testData.loc[testData['LotShape']=='Reg','isRegularShape'] = 1

    #Neighborhood Manipulation
    q1,q2,q3 = trainData['SalePrice'].quantile([0.25,0.5,0.75])
    #trainData['regionMedianPrice'] = 0
    #testData['regionMedianPrice'] = 0
    trainData['typeOfNeighborhood'] = 0
    testData['typeOfNeighborhood'] = 0
    neighborhoodList = []
    for name in trainData['Neighborhood']:
        if name not in neighborhoodList:
            neighborhoodList.append(name)
    neighborhoodList.sort()
    print(neighborhoodList)
    for name in neighborhoodList:
        median = trainData.loc[trainData['Neighborhood'] == name,'SalePrice'].median()
        #trainData.loc[trainData['Neighborhood'] == name,'regionMedianPrice'] = trainData.loc[trainData['Neighborhood'] == name,'SalePrice'].median()
        if median< q1:
            trainData.loc[trainData['Neighborhood'] == name,'typeOfNeighborhood'] = 0
            testData.loc[testData['Neighborhood'] == name,'typeOfNeighborhood'] = 0
        elif q1<= median <q2:
            trainData.loc[trainData['Neighborhood'] == name,'typeOfNeighborhood'] = 1
            testData.loc[testData['Neighborhood'] == name,'typeOfNeighborhood'] = 1
        elif q2<=median<q3:
            trainData.loc[trainData['Neighborhood'] == name,'typeOfNeighborhood'] = 2
            testData.loc[testData['Neighborhood'] == name,'typeOfNeighborhood'] = 2
        else:
            trainData.loc[trainData['Neighborhood'] == name,'typeOfNeighborhood'] = 3
            testData.loc[testData['Neighborhood'] == name,'typeOfNeighborhood'] = 3

    









    #neighhorMap = {'B':0,'C':1,'E':2,'G':3,'I':4,'M':5,'N':6,'O':7,'S':8,'T':9,'V':10}
    #trainData['NeighborhoodInNum'] = trainData['Neighborhood'].map(neighhorMap)
    #testData['NeighborhoodInNum'] = testData['Neighborhood'].map(neighhorMap)

    #Modeling

    y = trainData['SalePrice']
    feature = ['MSZoningNum','LotFrontage','LotArea','StreetPaved','isRegularShape']
    X = trainData[feature]

    print('It is working')
main()
