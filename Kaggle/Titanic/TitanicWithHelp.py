import os
import re
import string
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import SVC



from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def get_oof_proba(estimator, X, y, n_splits=5, seed=42):
    """对能 predict_proba 的分类器产生 OOF 概率（正类概率）"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in skf.split(X, y):
        est_ = clone(estimator)
        est_.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = est_.predict_proba(X.iloc[va_idx])[:, 1]
    return oof

def search_best_alpha_on_oof(oof_strong, oof_weak, y):
    """在 OOF 上搜最佳 alpha (强模型权重)，alpha ∈ [0.55, 0.95]"""
    alphas = np.linspace(0.55, 0.95, 9)
    best_alpha, best_acc = None, -1.0
    for a in alphas:
        blend = a * oof_strong + (1 - a) * oof_weak
        pred = (blend >= 0.5).astype(int)  # 阶段一先用 0.5
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc, best_alpha = acc, a
    return best_alpha, best_acc

def search_best_threshold_on_oof(proba, y):
    """在融合后的 OOF 概率上搜最佳阈值 t ∈ [0.30, 0.70]"""
    ts = np.linspace(0.30, 0.70, 41)
    best_t, best_acc = 0.5, -1.0
    for t in ts:
        acc = accuracy_score(y, (proba >= t).astype(int))
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc



def get_mae(maxLeafNode,train_X,val_X,train_y,val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=maxLeafNode,random_state=42)
    model.fit(train_X,train_y)
    output = model.predict(val_X)
    return mean_absolute_error(val_y,output)

def ticket_number(x):
    return x.split(' ')[-1]

def ticket_item(x):
    return x.split(' ')[0]

def main():

    trainFilePath = r"D:\Kaggle\Titanic\train.csv"
    testFilePath = r"D:\Kaggle\Titanic\test.csv"
    titanicTrainData = pd.read_csv(trainFilePath)
    titanicTestData = pd.read_csv(testFilePath)


    #Dealing P-class
    titanicTrainData['PClass'] = titanicTrainData['Pclass']
    titanicTestData['PClass'] = titanicTestData['Pclass']
    titanicTrainData = pd.get_dummies(titanicTrainData,columns=['Pclass'],prefix='pClass')
    titanicTestData = pd.get_dummies(titanicTestData,columns=['Pclass'],prefix='pClass')


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
    titanicTestData.loc[(titanicTestData['Age']<ageQ3) & (titanicTestData['Age']>=16),'is_Adults'] = 1

    
    titanicTrainData['is_Old'] = 0
    titanicTestData['is_Old'] = 0
    titanicTrainData.loc[(titanicTrainData['Age']>=ageQ3),'is_Old']=1
    titanicTestData.loc[(titanicTestData['Age']>=ageQ3),'is_Old']=1




    #Transform cabin
    #cabinMap ={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}
    titanicTrainData['CabinSection'] = titanicTrainData['Cabin'].str[0]
    titanicTestData['CabinSection'] = titanicTestData['Cabin'].str[0]
    titanicTrainData['CabinSection']=titanicTrainData['CabinSection'].fillna('M')
    titanicTestData['CabinSection']=titanicTestData['CabinSection'].fillna('M')
    cabinList = ['A','B','C','D','E', 'F']
    titanicTrainData['cabinEV'] = 0.0
    titanicTestData['cabinEV'] = 0.0
    for cabin in cabinList:
        mean = titanicTrainData.loc[titanicTrainData['CabinSection']==cabin, 'Survived'].mean()
        titanicTrainData.loc[titanicTrainData['CabinSection']==cabin,'cabinEV']=mean
        titanicTestData.loc[titanicTestData['CabinSection']==cabin,'cabinEV']=mean
    titanicTrainData['cabinEV'] = titanicTrainData['cabinEV'].fillna(0.0)
    titanicTestData['cabinEV']  = titanicTestData['cabinEV'].fillna(0.0)

    titanicTrainData['is_highClass'] = 0
    titanicTestData['is_highClass'] = 0
    titanicTrainData.loc[titanicTrainData['CabinSection']==('A'or'B'or'C'),'is_highClass'] = 1
    titanicTestData.loc[titanicTestData['CabinSection']==('A'or'B'or'C'),'is_highClass'] = 1
    titanicTrainData['is_missingClass']=0
    titanicTestData['is_missingClass']=0
    titanicTrainData.loc[titanicTrainData['CabinSection']=='M','is_missingClass']=1
    titanicTestData.loc[titanicTestData['CabinSection']=='M','is_missingClass']=1


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

    titanicTrainData['FarePerPerson'] = titanicTrainData['Fare'] / (titanicTrainData['familyNum']+1)
    titanicTestData['FarePerPerson']  = titanicTestData['Fare'] / (titanicTestData['familyNum']+1)

    






    #Dealing with ticket
    titanicTrainData['ticketNum'] = titanicTrainData['Ticket'].apply(ticket_number)
    titanicTestData['ticketNum'] = titanicTestData['Ticket'].apply(ticket_number)
    titanicTrainData['ticketNum'] = pd.to_numeric(titanicTrainData['ticketNum'], errors='coerce')
    titanicTestData['ticketNum']  = pd.to_numeric(titanicTestData['ticketNum'],  errors='coerce')
    med = titanicTrainData['ticketNum'].median()
    titanicTrainData['ticketNum'] = titanicTrainData['ticketNum'].fillna(med)
    titanicTestData['ticketNum']  = titanicTestData['ticketNum'].fillna(med)

    titanicTrainData['ticketFirst'] = titanicTrainData['Ticket'].str[0]
    titanicTestData['ticketFirst'] = titanicTestData['Ticket'].str[0]
    titanicTrainData['haveTicketItem'] = 0
    titanicTestData['haveTicketItem'] = 0

    titanicTrainData.loc[~titanicTrainData['ticketFirst'].str.isdigit(),'haveTicketItem'] = 1
    titanicTestData.loc[~titanicTestData['ticketFirst'].str.isdigit(),'haveTicketItem'] =1


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



    #Is a mother?
    titanicTrainData['isMother']=0
    titanicTestData['isMother']=0
    titanicTrainData.loc[(titanicTrainData['Age']>18) & (titanicTrainData['genderInNum']==1) & (titanicTrainData['is_Miss']==0) & (titanicTrainData['familyNum']>1),
                         'isMother'] =1
    titanicTestData.loc[(titanicTestData['Age']>18) & (titanicTestData['genderInNum']==1) & (titanicTestData['is_Miss']==0) & (titanicTestData['familyNum']>1),
                         'isMother'] =1
    
    titanicTrainData['Child_Pclass'] = titanicTrainData['is_Child'] * titanicTrainData['PClass']
    titanicTestData['Child_Pclass']  = titanicTestData['is_Child']  * titanicTestData['PClass'] 




    print(titanicTrainData.columns)



    #Fulfill the empty part in CabinSectionIndex with the median
    #titanicTrainData['CabinSectionIndex'] = titanicTrainData['CabinSectionIndex'].fillna(titanicTrainData['CabinSectionIndex'].median())
    #titanicTestData['CabinSectionIndex'] = titanicTestData['CabinSectionIndex'].fillna(titanicTestData['CabinSectionIndex'].median())
    #Choosing variable to use
    y = titanicTrainData['Survived']
    feature_Lin = ['pClass_1','pClass_2','pClass_3','genderInNum','embarkC','embarkQ',
               'embarkS','FarePerPerson','is_Child','is_Adults','is_Old','isAlone',
               'cabinEV', 'is_Master','is_Miss', 'is_Mr', 'is_Mrs','is_Rare','ticketNum','Child_Pclass']
    feature_Tree= ['PClass','genderInNum','embarkC','embarkQ','embarkS','FarePerPerson','is_Child','is_Adults','is_Old','isAlone','cabinEV', 'is_Master','is_Miss', 'is_Mr', 'is_Mrs','is_Rare','ticketNum']
    X_Lin = titanicTrainData[feature_Lin]
    X_Tree = titanicTrainData[feature_Tree]


    #Validation
    train_X_Lin,val_X_Lin,train_y_Lin,val_y_Lin=train_test_split(X_Lin,y,random_state=0)
    train_X_Tree,val_X_Tree,train_y_Tree,val_y_Tree=train_test_split(X_Tree,y,random_state=0)





    #logistic regression
    logreg = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    logreg.fit(train_X_Lin, train_y_Lin)
    val_pred_lr = logreg.predict(val_X_Lin)
    print("LogReg Validation Accuracy:", accuracy_score(val_y_Lin, val_pred_lr))

    # ===== Random Forest（保留原来）=====
    rf = RandomForestClassifier(n_estimators=400,
                                max_depth=15,
                                min_samples_leaf=1,
                                random_state=42)
    rf.fit(train_X_Tree, train_y_Tree)
    val_pred_rf = rf.predict(val_X_Tree)
    print("RandomForest Validation Accuracy:", accuracy_score(val_y_Tree, val_pred_rf))

    #Xgboost
    xgb = XGBClassifier(
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric="logloss",
    )

    xgb.fit(train_X_Tree, train_y_Tree)
    val_pred_xgb = xgb.predict(val_X_Tree)
    print("XGBoost Validation Accuracy:", accuracy_score(val_y_Tree, val_pred_xgb))



    # 用全部训练数据拟合并生成两份提交：RF 与 LogReg
    # RF
    rf.fit(X_Tree, y)
    pred_rf = rf.predict(titanicTestData[feature_Tree])
    submission_rf = pd.DataFrame({'PassengerId':titanicTestData['PassengerId'],'Survived':pred_rf})
    submission_rf.to_csv('submission_rf.csv', index=False)

    # LogReg
    logreg.fit(X_Lin, y)
    pred_lr = logreg.predict(titanicTestData[feature_Lin])
    submission_lr = pd.DataFrame({'PassengerId':titanicTestData['PassengerId'],'Survived':pred_lr})
    submission_lr.to_csv('submission_logreg.csv', index=False)

    # XGBoost
    xgb.fit(X_Tree, y)
    pred_xgb = xgb.predict(titanicTestData[feature_Tree])
    submission_xgb = pd.DataFrame({'PassengerId': titanicTestData['PassengerId'], 'Survived': pred_xgb})
    submission_xgb.to_csv('submission_xgb.csv', index=False)



    # 1) 为两模型生成 OOF 概率（注意：LogReg 用 X_Lin，XGB 用 X_Tree）
    oof_xgb = get_oof_proba(xgb, X_Tree, y, n_splits=5, seed=42)
    oof_lr  = get_oof_proba(logreg, X_Lin, y, n_splits=5, seed=42)

    # 2) 概率相关性（越低越互补）
    print("概率相关性:", np.corrcoef(oof_xgb, oof_lr)[0, 1])

    # 3) 在 OOF 上先搜最佳 alpha（给 XGB 的权重）
    best_alpha, cv_acc_alpha = search_best_alpha_on_oof(oof_xgb, oof_lr, y)
    print(f"[Blend] best alpha = {best_alpha:.3f}, CV acc (0.5 thr.) = {cv_acc_alpha:.5f}")

    # 4) 在 OOF 上再搜最佳阈值 t
    blend_oof = best_alpha * oof_xgb + (1 - best_alpha) * oof_lr
    best_t, cv_acc_t = search_best_threshold_on_oof(blend_oof, y)
    print(f"[Blend] best threshold t = {best_t:.3f}, CV acc (with t) = {cv_acc_t:.5f}")

    # 5) 全量重训两模型，拿测试集概率
    xgb_full   = clone(xgb).fit(X_Tree, y)
    logreg_full= clone(logreg).fit(X_Lin, y)
    proba_xgb_test = xgb_full.predict_proba(titanicTestData[feature_Tree])[:, 1]
    proba_lr_test  = logreg_full.predict_proba(titanicTestData[feature_Lin])[:, 1]

    # 6) 用最佳 alpha、t 融合生成提交
    blend_test = best_alpha * proba_xgb_test + (1 - best_alpha) * proba_lr_test
    final_pred_blend = (blend_test >= best_t).astype(int)
    submission_blend = pd.DataFrame({'PassengerId': titanicTestData['PassengerId'], 'Survived': final_pred_blend})
    submission_blend.to_csv('submission_blend_xgb_lr.csv', index=False)
    print("已生成: submission_blend_xgb_lr.csv")
    corr = np.corrcoef(pred_xgb, pred_lr)[0, 1]
    print("预测相关性:", corr)
    

    print(f"当前的工作目录是: {os.getcwd()}")
    print("已生成: submission_rf.csv, submission_logreg.csv,submission_xgb")
main()