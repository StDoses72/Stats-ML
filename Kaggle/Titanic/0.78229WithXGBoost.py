import os
import re
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import SVC



def get_mae(maxLeafNode,train_X,val_X,train_y,val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=maxLeafNode,random_state=42)
    model.fit(train_X,train_y)
    output = model.predict(val_X)
    return mean_absolute_error(val_y,output)

def ticket_number(x):
    return x.split(' ')[-1]

def main():

    trainFilePath = r"D:\Kaggle\Titanic\train.csv"
    testFilePath = r"D:\Kaggle\Titanic\test.csv"
    titanicTrainData = pd.read_csv(trainFilePath)
    titanicTestData = pd.read_csv(testFilePath)
    print(titanicTrainData.columns)



    #Change string in sex into int
    sexMap = {'male':0,'female':1}
    titanicTrainData['genderInNum'] = titanicTrainData['Sex'].map(sexMap)
    titanicTestData['genderInNum'] = titanicTestData['Sex'].map(sexMap)

    #Calculate the number of Family
    titanicTrainData['familyNum'] = titanicTrainData['SibSp']+titanicTrainData['Parch']
    titanicTestData['familyNum'] = titanicTestData['SibSp']+titanicTestData['Parch']

    #Changing number of family into whether single
    titanicTrainData['isAlone'] = 0
    titanicTestData['isAlone'] =0
    titanicTrainData.loc[titanicTrainData['familyNum']==0,'isAlone']=1#.loc(row_condition,newColumn)=sss assign value ss to new column with previous row condition of row_condition
    titanicTestData.loc[titanicTestData['familyNum']==0,'isAlone']=1

    #Change Embark in C,Q,S into int
    
    embarkList = ['C','Q','S']
    for embark in embarkList:
        titanicTrainData['embark'+embark] = 0
        titanicTestData['embark'+embark] = 0
        titanicTrainData.loc[titanicTrainData['Embarked']==embark,'embark'+embark]=1
        titanicTestData.loc[titanicTestData['Embarked']==embark,'embark'+embark]=1


    #embarkmap = {'C':0,'Q':1,'S':2}
    #titanicTrainData['Embarked'] = titanicTrainData['Embarked'].fillna(titanicTrainData['Embarked'].mode())
    #titanicTestData['Embarked'] = titanicTestData['Embarked'].fillna(titanicTrainData['Embarked'].mode())
    #titanicTrainData['embarkInNum'] = titanicTrainData['Embarked'].map(embarkmap)
    #titanicTestData['embarkInNum'] = titanicTestData['Embarked'].map(embarkmap)


    #Fulfilling empty space in Age with median
    ageQ3 = titanicTrainData['Age'].quantile(0.75)
    titanicTrainData['Age'] = titanicTrainData['Age'].fillna(titanicTrainData['Age'].median())
    titanicTestData['Age'] = titanicTestData['Age'].fillna(titanicTrainData['Age'].median())

    titanicTrainData['is_Child'] = 0
    titanicTestData['is_Child'] = 0

    titanicTrainData.loc[titanicTrainData['Age']<16,'is_Child'] = 1
    titanicTestData.loc[titanicTestData['Age']<16,'is_Child'] = 1

    titanicTrainData['is_Adults'] = 0
    titanicTestData['is_Adults'] = 0

    titanicTrainData.loc[(titanicTrainData['Age']<ageQ3) & (titanicTrainData['Age']>=16),'is_Adults'] = 1
    titanicTestData.loc[(titanicTestData['Age']<ageQ3) & (titanicTrainData['Age']>=16),'is_Adults'] = 1

    
    titanicTrainData['is_Old'] = 0
    titanicTestData['is_Old'] = 0
    titanicTrainData.loc[(titanicTrainData['Age']>=ageQ3),'is_Old']=1
    titanicTestData.loc[(titanicTestData['Age']>=ageQ3),'is_Old']=1




    #Transform cabin
    #cabinMap ={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}
    titanicTrainData['CabinSection'] = titanicTrainData['Cabin'].str[0]
    titanicTestData['CabinSection'] = titanicTestData['Cabin'].str[0]
    cabinList = ['A','B','C','D','E', 'F']
    titanicTrainData['cabinEV'] = 0.0
    titanicTestData['cabinEV'] = 0.0
    for cabin in cabinList:
        mean = titanicTrainData.loc[titanicTrainData['CabinSection']==cabin, 'Survived'].mean()
        titanicTrainData.loc[titanicTrainData['CabinSection']==cabin,'cabinEV']=mean
        titanicTestData.loc[titanicTestData['CabinSection']==cabin,'cabinEV']=mean
    titanicTrainData['cabinEV'] = titanicTrainData['cabinEV'].fillna(0.0)
    titanicTestData['cabinEV']  = titanicTestData['cabinEV'].fillna(0.0)


    #for cabin in cabinList:
        #titanicTrainData['is'+str(cabin)] = 0
        #titanicTestData['is'+str(cabin)] = 0
        #titanicTrainData.loc[titanicTrainData['CabinSection']==cabin, 'is'+str(cabin)] = 1
        #titanicTestData.loc[titanicTestData['CabinSection']==cabin, 'is'+str(cabin)] = 1



    #titanicTrainData['CabinSectionIndex'] = titanicTrainData['CabinSection'].map(cabinMap)
    #titanicTestData['CabinSectionIndex'] = titanicTestData['CabinSection'].map(cabinMap)

    #Fill missing Fare
    titanicTrainData['Fare']= titanicTrainData['Fare'].fillna(titanicTrainData['Fare'].median())
    titanicTestData['Fare']=titanicTestData['Fare'].fillna(titanicTrainData['Fare'].median())

    titanicTrainData['logFare'] = np.log1p(titanicTrainData['Fare'])
    titanicTestData['logFare'] = np.log1p(titanicTestData['Fare'])

    titanicTrainData['logFare']=titanicTrainData['logFare'].fillna(titanicTrainData['logFare'].median())
    titanicTestData['logFare']=titanicTestData['logFare'].fillna(titanicTrainData['logFare'].median())









    #Dealing with ticket
    titanicTrainData['ticketNum'] = titanicTrainData['Ticket'].apply(ticket_number)
    titanicTestData['ticketNum'] = titanicTestData['Ticket'].apply(ticket_number)
    titanicTrainData['ticketNum'] = pd.to_numeric(titanicTrainData['ticketNum'], errors='coerce')
    titanicTestData['ticketNum']  = pd.to_numeric(titanicTestData['ticketNum'],  errors='coerce')
    med = titanicTrainData['ticketNum'].median()
    titanicTrainData['ticketNum'] = titanicTrainData['ticketNum'].fillna(med)
    titanicTestData['ticketNum']  = titanicTestData['ticketNum'].fillna(med)


    #处理Name
    titleMap = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",

    # everything else goes to Rare
    "Don": "Rare",
    "Rev": "Rare",
    "Dr": "Rare",
    "Mme": "Rare",
    "Ms": "Rare",
    "Major": "Rare",
    "Lady": "Rare",
    "Sir": "Rare",
    "Mlle": "Rare",
    "Col": "Rare",
    "Capt": "Rare",
    "the Countess": "Rare",
    "Jonkheer": "Rare"
}
    titanicTrainData['Title'] = titanicTrainData['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    titanicTestData['Title'] = titanicTestData['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    titanicTrainData['finalTitle'] = titanicTrainData['Title'].map(titleMap)
    titanicTestData['finalTitle'] = titanicTestData['Title'].map(titleMap)
    finalTitleList = ['Rare','Miss','Mrs','Mr','Master']
    for title in finalTitleList:
        titanicTrainData['is_'+title] = 0
        titanicTestData['is_'+title] =  0
        titanicTrainData.loc[titanicTrainData['finalTitle']==title,'is_'+title] = 1
        titanicTestData.loc[titanicTestData['finalTitle']==title,'is_'+title] = 1


    print(titanicTrainData.columns)


    #Fulfill the empty part in CabinSectionIndex with the median
    #titanicTrainData['CabinSectionIndex'] = titanicTrainData['CabinSectionIndex'].fillna(titanicTrainData['CabinSectionIndex'].median())
    #titanicTestData['CabinSectionIndex'] = titanicTestData['CabinSectionIndex'].fillna(titanicTestData['CabinSectionIndex'].median())
    #Choosing variable to use
    y = titanicTrainData['Survived']
    feature = ['Pclass','genderInNum','embarkC','embarkQ','embarkS','logFare','is_Child','is_Adults','is_Old','isAlone','cabinEV', 'is_Master','is_Miss', 'is_Mr', 'is_Mrs','is_Rare','ticketNum']
    X = titanicTrainData[feature]


    #Validation
    train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=0)


    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm.fit(train_X, train_y)

    val_pred_svm = svm.predict(val_X)
    print("SVM Validation Accuracy:", accuracy_score(val_y, val_pred_svm))

    #logistic regression
    logreg = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    logreg.fit(train_X, train_y)
    val_pred_lr = logreg.predict(val_X)
    print("LogReg Validation Accuracy:", accuracy_score(val_y, val_pred_lr))

    # ===== Random Forest（保留原来）=====
    rf = RandomForestClassifier(n_estimators=350,
                                max_depth=15,
                                min_samples_leaf=1,
                                random_state=42)
    rf.fit(train_X, train_y)
    val_pred_rf = rf.predict(val_X)
    print("RandomForest Validation Accuracy:", accuracy_score(val_y, val_pred_rf))

    #Xgboost
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.09,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    xgb.fit(train_X, train_y)
    val_pred_xgb = xgb.predict(val_X)
    print("XGBoost Validation Accuracy:", accuracy_score(val_y, val_pred_xgb))

    

    # 用全部训练数据拟合并生成两份提交：RF 与 LogReg
    # RF
    rf.fit(X, y)
    pred_rf = rf.predict(titanicTestData[feature])
    submission_rf = pd.DataFrame({'PassengerId':titanicTestData['PassengerId'],'Survived':pred_rf})
    submission_rf.to_csv('submission_rf.csv', index=False)

    # LogReg
    logreg.fit(X, y)
    pred_lr = logreg.predict(titanicTestData[feature])
    submission_lr = pd.DataFrame({'PassengerId':titanicTestData['PassengerId'],'Survived':pred_lr})
    submission_lr.to_csv('submission_logreg.csv', index=False)

    # XGBoost
    xgb.fit(X, y)
    pred_xgb = xgb.predict(titanicTestData[feature])
    submission_xgb = pd.DataFrame({'PassengerId': titanicTestData['PassengerId'], 'Survived': pred_xgb})
    submission_xgb.to_csv('submission_xgb.csv', index=False)

    

    print(f"当前的工作目录是: {os.getcwd()}")
    print("已生成: submission_rf.csv, submission_logreg.csv,submission_xgb")
main()
