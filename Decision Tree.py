#!/usr/bin/env python
# coding: utf-8

# In[52]:


import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[174]:


df=pd.read_excel(r"C:\Users\kisho\Downloads\bank.xlsx",sheet_name="bank")


# In[175]:


df.head()


# In[176]:


df.info()


# In[177]:


df["job"].describe()


# In[178]:


df["job"].value_counts()


# In[179]:


df["job"]=df["job"].replace(["admin.","services","housemaid"],"pink_collar")


# In[180]:


df["job"]=df["job"].replace(["retired","student","unemployed"],"dependents")


# In[181]:


df["job"]=df["job"].replace(["self-employed","entrepreneur"],"self_employed")


# In[182]:


df["job"].value_counts()


# In[183]:


df["default"].value_counts()


# In[184]:


df["default"]=df["default"].map({"yes":1,"no":0})


# In[185]:


df["default"].value_counts()


# In[186]:


df.drop(columns="contact",inplace=True) #remove contact, day, month


# In[187]:


df.drop(columns="day",inplace=True)


# In[188]:


df.drop(columns="month",inplace=True)


# In[189]:


df["housing"].value_counts()


# In[190]:


df["housing"]=df["housing"].map({"yes":1,"no":0})


# In[191]:


df["poutcome"].value_counts() #unknown and other be merged in one


# In[192]:


df["poutcome"]=df["poutcome"].replace(["unknown","other"],"other")


# In[193]:


df["poutcome"].value_counts()


# In[194]:


df["deposit"].value_counts() #yes and no as 1 and 0


# In[195]:


df["deposit"]=df["deposit"].map({"yes":1,"no":0})


# In[198]:


df["loan"].value_counts()


# In[197]:


df["loan"]=df["loan"].map({"yes":1,"no":0})


# In[199]:


df.info()


# In[200]:


df["pdays"].value_counts() 


# In[201]:


df["pdays"]=df["pdays"].replace([-1],999)


# In[202]:


df["pdays"]=1/(df["pdays"])


# In[203]:


df.describe(percentiles=[.01,.02,.03,.04,.05,.1,.5,.75,.90,.95,]).T


# # Numerical

# In[204]:


num_data=df[df.dtypes[df.dtypes!="object"].index]
cat_data=df[df.dtypes[df.dtypes=="object"].index]


# In[205]:


num_data


# In[206]:


def outlier_cap_box_plot(var):
    q1=var.quantile(.01)
    q3=var.quantile(.99)
    iqr=q3-q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    var=np.where(var>=upper_bound,upper_bound,var)
    var=np.where(var<=lower_bound,lower_bound,var)
    return var


# In[207]:


num_data2=num_data.apply(outlier_cap_box_plot)


# In[208]:


num_data2.describe(percentiles=[.01,.02,.03,.04,.05,.1,.5,.75,.90,.95,]).T


# In[209]:


df=pd.concat([cat_data,num_data2],axis=1)


# In[210]:


df.head(5)


# In[211]:


df1=pd.get_dummies(data=df,drop_first=True)


# In[212]:


df1.head(10)


# In[213]:


df1.isnull().sum()


# In[214]:


df1.columns


# In[229]:


y=df1["deposit"]
x=df1[['age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign',
       'pdays', 'previous', 'job_dependents', 'job_management',
       'job_pink_collar', 'job_self_employed', 'job_technician', 'job_unknown',
       'marital_married', 'marital_single', 'education_secondary',
       'education_tertiary', 'education_unknown', 'poutcome_other',
       'poutcome_success']]


# In[230]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=88)


# In[231]:


from sklearn.linear_model import LogisticRegression
logistic_model=LogisticRegression(max_iter=1000)


# In[232]:


logistic_model.fit(x_train,y_train)


# In[233]:


logistic_model.score(x_train,y_train)


# In[234]:


logistic_model.score(x_test,y_test)


# In[236]:


pred_train=logistic_model.predict(x_train)


# In[237]:


pred_test=logistic_model.predict(x_test)


# In[ ]:


from sklearn import metrics


# In[238]:


cmtrain=metrics.confusion_matrix(y_train,pred_train)
pd.DataFrame(cmtrain,columns=["pred_0","pred_1"],index=["act_0","act_1"])


# In[239]:


print(metrics.classification_report(y_train,pred_train))


# In[240]:


cmtest=metrics.confusion_matrix(y_test,pred_test)
pd.DataFrame(cmtest,columns=["pred_0","pred_1"],index=["act_0","act_1"])


# In[241]:


print(metrics.classification_report(y_test,pred_test))


# In[242]:


dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Train Accuracy :", round(dt.score(x_train,y_train),3))
print("Test Accuracy :",round(dt.score(x_test,y_test),3))


# In[252]:


dt1=DecisionTreeClassifier(max_depth=7)
dt1.fit(x_train,y_train)
print("Train Accuracy :", round(dt1.score(x_train,y_train),3))
print("Test Accuracy :",round(dt1.score(x_test,y_test),3))


# In[258]:


dt1=DecisionTreeClassifier(min_samples_split=300)
dt1.fit(x_train,y_train)
print("Train Accuracy :", (round(dt1.score(x_train,y_train),3)*100))
print("Test Accuracy :", (round(dt1.score(x_test,y_test),3)*100))


# # Grid Search

# In[261]:


from sklearn.model_selection import GridSearchCV
params={"criterion":["gini","entropy"],
       "max_depth":[5,7,9,10,11],
       "min_samples_split":[10,15,20,50,100,200,250],
       "min_samples_leaf":[5,10,15,20,50,80,100]}
dtg=DecisionTreeClassifier(random_state=0)
gd_search=GridSearchCV(estimator=dtg,param_grid=params,cv=10,n_jobs=2)
gd_search.fit(x_train,y_train) #doubt

n_jobs=-1 (means use all logical processor)
n_jobs=1 (means use only 1 logical processor)
n_jobs=2 (means use only 2 logical processor)
n_jobs=3 (means use only 3 logical processor)
# In[263]:


gd_search.best_params_


# In[264]:


gd_search.best_score_


# In[266]:


gd_search.best_estimator_


# In[267]:


dt_f=DecisionTreeClassifier(criterion="gini",max_depth=10,min_samples_leaf=20,min_samples_split=250,random_state=0)
dt_f.fit(x_train,y_train)


# In[268]:


print("Train Accuracy:" , dt_f.score(x_train,y_train))
print("Test Accuracy:", dt_f.score(x_test,y_test))


# In[270]:


from sklearn.tree import plot_tree
fn=x_train.columns
cn=["yes","no"]

#setting dpi=300 to make image clearer than default
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(5,5),dpi=500)

dt_plot=plot_tree(dt_f,feature_names=fn,class_names=cn,filled=True);


# # Feature Importance

# In[271]:


dt_f.feature_importances_


# In[273]:


pd.DataFrame({"Features":x_train.columns,"Importance":dt_f.feature_importances_}).sort_values(by=["Importance"],ascending = False)


# In[ ]:




