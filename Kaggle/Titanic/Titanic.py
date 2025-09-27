import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split



def get_mae(maxLeafNode,train_X,val_X,train_y,val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=maxLeafNode,random_state=42)
    model.fit(train_X,train_y)
    output = model.predict(val_X)
    return mean_absolute_error(val_y,output)


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
    embarkmap = {'C':0,'Q':1,'S':2}
    titanicTrainData['Embarked'] = titanicTrainData['Embarked'].fillna(titanicTrainData['Embarked'].mode())
    titanicTestData['Embarked'] = titanicTestData['Embarked'].fillna(titanicTrainData['Embarked'].mode())
    titanicTrainData['embarkInNum'] = titanicTrainData['Embarked'].map(embarkmap)
    titanicTestData['embarkInNum'] = titanicTestData['Embarked'].map(embarkmap)


    #Fulfilling empty space in Age with median
    titanicTrainData['Age'] = titanicTrainData['Age'].fillna(titanicTrainData['Age'].median())
    titanicTestData['Age'] = titanicTestData['Age'].fillna(titanicTrainData['Age'].median())

    #Transform cabin into string, then into number
    cabinMap ={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}
    titanicTrainData['CabinSection'] = titanicTrainData['Cabin'].str[0]
    titanicTestData['CabinSection'] = titanicTestData['Cabin'].str[0]
    titanicTrainData['CabinSectionIndex'] = titanicTrainData['CabinSection'].map(cabinMap)
    titanicTestData['CabinSectionIndex'] = titanicTestData['CabinSection'].map(cabinMap)

    #Fill missing Fare
    titanicTrainData['Fare'].fillna(titanicTrainData['Fare'].median(),inplace=True)
    titanicTestData['Fare'].fillna(titanicTrainData['Fare'].median(),inplace=True)

    #Fulfill the empty part in CabinSectionIndex with the median
    titanicTrainData['CabinSectionIndex'] = titanicTrainData['CabinSectionIndex'].fillna(titanicTrainData['CabinSectionIndex'].median())
    titanicTestData['CabinSectionIndex'] = titanicTestData['CabinSectionIndex'].fillna(titanicTestData['CabinSectionIndex'].median())
    #Choosing variable to use
    y = titanicTrainData['Survived']
    feature = ['Pclass','genderInNum','embarkInNum','Fare','Age','CabinSectionIndex','isAlone']
    X = titanicTrainData[feature]


    #Validation
    train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=0)
    for maxLeafNode in [65,66,67,68]:
        print(maxLeafNode,get_mae(maxLeafNode,train_X,val_X,train_y,val_y))


    
    #Modeling
    titanicModel = DecisionTreeClassifier(max_leaf_nodes=65,random_state=42)
    #titanicModel = RandomForestClassifier(random_state=42)

    titanicModel.fit(X, y)
    prediction = titanicModel.predict(titanicTestData[feature])
    print(prediction)

    
    




    submissionDataFrame = pd.DataFrame({'PassengerId':titanicTestData['PassengerId'],'Survived':prediction})
    submissionDataFrame.to_csv('submission.csv', index=False)
    print(f"当前的工作目录是: {os.getcwd()}")

main()
