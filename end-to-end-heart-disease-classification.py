#!/usr/bin/env python
# coding: utf-8

# In[81]:


#preparing the tools
# regular Exploratory Data Analysis (EDA) and plotting libraries
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

#to make our plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

## Model evaluators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# ## Load Data

# In[82]:


df = pd.read_csv("heart-disease.csv")
df.shape


# #exploratory data analysis
# #find more abt the data, and become an expert on the subjrct matter of the dataset
# 1. What question(s) are you trying to solve (or prove wrong)?
# 2. What kind of data do you have and how do you treat different types?
# 3. What’s missing from the data and how do you deal with it?
# 4. Where are the outliers and why should you care about them?
# 5. How can you add, change or remove features to get more out of your data?

# In[83]:


df.head()


# In[84]:


df.tail()


# In[85]:


df['target'].value_counts()
#165 have the disease 138 do not


# In[86]:


df['target'].value_counts().plot(kind="bar", 
                                 color=["salmon", "lightblue"]);


# In[87]:


df.info()


# In[88]:


#check for missng values
df.isna().sum()


# In[89]:


df.describe()


# Heart Disease Frequency according to Gender
# If you want to compare two columns to each other, you can use the function pd.crosstab(column_1, column_2).
# 
# This is helpful if you want to start gaining an intuition about how your independent variables interact with your dependent variables.
# 
# Let's compare our target column with the sex column.
# 
# Remember from our data dictionary, for the target column, 1 = heart disease present, 0 = no heart disease. And for sex, 1 = male, 0 = femal

# In[90]:


df.sex.value_counts()
# 1= male
# 2= female


# In[91]:


# Compare target column with sex column
pd.crosstab(df.target, df.sex,)


# In[92]:


#create a plot of crosstab
pd.crosstab(df.target, df.sex,).plot(kind='bar', 
                                     figsize=(10,6) ,
                                     color=['salmon','lightblue'])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('0= No Diseease, 1= Disease')
plt.ylabel('Amount' )
plt.legend(['Female', 'Male'])
plt.xticks(rotation=0)


# In[93]:


df.head()


# In[94]:


df['chol'].value_counts()
#length shows the number of different values in a column for the chol colum, there are 152 different values 
# and as such the variations are too much to plot on a bar chat


# ##  Age vs Max Heart Rate (thalach) vs Target

# In[95]:


#create another figure
plt.figure(figsize=(10,6))

#with heart disease
plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
           color="salmon")

#withouth heart disease
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
           color="lightblue");
# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Max Heart Rate");


# In[96]:


#check the distribution of the age column with the histogram
df.age.plot.hist();


# ## Heart Disease Frequency per Chest Pain Type¶
# 
#  0: Typical angina: chest pain related decrease blood supply to the heart
#  
#  1: Atypical angina: chest pain not related to heart
#  
#  2: Non-anginal pain: typically esophageal spasms (non heart related)
#  
#  3: Asymptomatic: chest pain not showing signs of disease

# In[97]:


pd.crosstab(df.cp, df.target)


# In[98]:


pd.crosstab(df.cp,df.target).plot(kind='bar',
                                   figsize=(10,6),
                                   color=["salmon","lightblue"]);
                                  


# In[99]:


#make a correlation matrix
df.corr()


# In[100]:


#making our correlation more visisble with seaborn sea map
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,
                annot=True,
                linewidth= 0.5,
                fmt='.2f',
                cmap="YlGnBu")


# In[101]:


pd.crosstab(df.target, df.exang).plot(kind='bar', 
                                     figsize=(10,6) ,
                                     color=['salmon','lightblue'])
plt.title('Heart Disease Frequency for Exercise Inducesd Angina')
plt.xlabel('0= No Diseease, 1= Disease')
plt.ylabel('Amount' )
plt.legend(['Female', 'Male'])
plt.xticks(rotation=0)


# ## Modelling

# In[102]:


df.head()


# In[103]:


#split data into x and y
x= df.drop('target', axis=1)
y= df['target']


# In[104]:


x


# In[105]:


y


# In[106]:


#split data into train and test
np.random.seed(42)
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                test_size=0.2)


# In[107]:


x_train


# In[108]:


y_train,len(y_train)


# # After spliting data build a machine learning model, sampling three different models
# 1. Logistic regression
# 2. k Neighbors classifiers
# 3. Random Forest Classifiers

# In[109]:


# Put models in a dictionary
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(), 
          "Random Forest": RandomForestClassifier()}

# Create function to fit and score models
def fit_and_score(models, x_train, x_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(x_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(x_test, y_test)
    return model_scores


# In[110]:


model_scores = fit_and_score(models=models,
                            x_train=x_train,
                            x_test= x_test,
                            y_train= y_train,
                            y_test= y_test)
model_scores


# In[111]:


model_compare = pd.DataFrame(model_scores,index=['Accuracy'])
model_compare.T.plot.bar();
 


# ##  Hyperparameter Tuning by Hand

# In[112]:


#tuning KNN bY hand

train_scores = []
test_scores =[]
#list of different values of n_neighbors
neighbors = range(1,21)

#setup KNN instance
knn = KNeighborsClassifier()

#loop through different neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    #fit the algorithm
    knn.fit(x_train, y_train)
    
    #update the training score list
    train_scores.append(knn.score(x_train ,y_train))
    
    #update the test scores list
    test_scores.append(knn.score(x_test,y_test))


# In[113]:


train_scores


# In[114]:


test_scores


# In[115]:


plt.plot(neighbors, train_scores, label='Train score')
plt.plot(neighbors, test_scores, label='Test score')
plt.xticks(np.arange(1,21,1))
plt.xlabel('Number of neigbors')
plt.ylabel('Model score')
plt.legend()

print(f'Maximum KNN score on the test data:{max(test_scores)*100:2f}')


# ## Hyperparameter Tuning with RandomizedSearchCV
# 
# #we will be tuning
# 1. Logistic Regression()
# 
# 2. Random ForestClassifier()

# In[116]:


# create a grid for logistic regression
log_reg_grid = {'C': np.logspace (-4,4,20),
               "solver":["liblinear"]}
#create hyperparameter grid for RandomSearchClassifier
rf_grid = {'n_estimators':np.arange(10,1000,50),
           "max_depth":[None,3,5,10],
           "min_samples_split":np.arange(2,20,2),
           "min_samples_leaf":np.arange(1,20,2)
          }
#number 10 to number 1000, 50 numbers apart


# In[117]:


# hyperparameter tuning of  logistic regression with RandomizedSearchCV 
np.random.seed(42)
rs_log_reg = RandomizedSearchCV(LogisticRegression()
                               , param_distributions=log_reg_grid,
                               cv=5,
                                n_iter=20,
                               verbose=True )
rs_log_reg.fit(x_train, y_train)


# In[118]:


rs_log_reg.best_params_


# In[119]:


rs_log_reg.score(x_test,y_test)


# In[120]:


np.random.seed(42)
rs_rf = RandomizedSearchCV(RandomForestClassifier()
                               , param_distributions=rf_grid,
                               cv=5,
                                n_iter=20,
                               verbose=True )
rs_rf.fit(x_train, y_train)


# In[121]:


rs_rf.best_params_


# In[122]:


rs_rf.score(x_test,y_test)


# ## Hyperparameter Tuning With GridSearchCV
# 
# 
# # since logistic regression is the best performing model of our last tuning , we would experiment on improving the score with GridSearchCV

# In[123]:


log_reg_grid = {'C': np.logspace (-4,4,30),
               "solver":["liblinear"]}

gs_log_reg = GridSearchCV(LogisticRegression()
                               , param_grid=log_reg_grid,
                               cv=5,
                               verbose=True )
gs_log_reg.fit(x_train, y_train)


# In[124]:


gs_log_reg.best_params_


# In[125]:


gs_log_reg.score(x_test, y_test)


# ##Evaluating our tuned machine learning classifier beyond accuracy
# 
# 
# ROC curve and AOC curve
# Confusin matrix
# Classification report
# Precision
# Recall
# F1 score
# 
#  make predictions to evaluate our model ##

# In[126]:


y_preds = gs_log_reg.predict(x_test)


# In[127]:


y_preds


# In[128]:


y_test


# In[129]:


#plot roc curve and calculate auc metric
plot_roc_curve(gs_log_reg, x_test, y_test)


# In[130]:


#confusion matrix
print(confusion_matrix(y_test, y_preds))


# In[131]:


sns


# In[132]:


sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """
    plot a confusion matrix using seaborn's heatmap
    """ 
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                    annot=True,
                    cbar=False)
    plt.xlabel("True Label")
    plt.ylabel("Predicted label")
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom +0.5, top - 0.5)

   
plot_conf_mat(y_test, y_preds)
    


# In[133]:


#Classification report and cross validation, recall and f1 score


# In[134]:


print(classification_report(y_preds, y_test))


# # calculate evaluation  matricx using cross validation (cross_val_score) : precision, f1 score, recall

# In[135]:


#check best hyperparameters to use for cross validation
gs_log_reg.best_params_


# In[136]:


#create a new classifier with the best parameter
clf = LogisticRegression(C = 0.20433597178569418, solver = 'liblinear')


# In[137]:


#cross validated accuracy

cv_acc = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring = "accuracy"
                         
)
cv_acc


# In[138]:


cv_acc = np.mean(cv_acc)
cv_acc


# In[139]:


#cross validated precision
cv_precision = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring = "precision"
                         
)
cv_precision = np.mean(cv_precision)
cv_precision


# In[140]:


#cross validated recalln
cv_recall = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring = "recall"
                         
)
cv_recall = np.mean(cv_recall)
cv_recall


# In[141]:


#cross validated f1_score
cv_f1 = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring = "f1"
                         
)
cv_f1 = np.mean(cv_f1)
cv_f1


# In[142]:


cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall": cv_recall,
                           "F1": cv_f1},
                         index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                     legend= False);                                    


# #Feature importance
# 
# This analysizes which of the features is the most important in processign the data.
# 
# The important features can be detected with different methods depending on the machine learning model
# 
# Finding the feature importance of our logistic regression model we would be using the "(MODEL NAME)"

# In[143]:


clf = LogisticRegression(C= 0.20433597178569418,
                       solver ="liblinear" )
clf.fit(x_train,y_train);


# In[144]:


clf.coef_


# In[145]:


#Match coefficient(coef) of features to their corresponding columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[146]:


#Another way to visualise a feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title = "Feature Importance", legend =False);


# In[147]:


pd.crosstab(df["slope"], df['target'])


# slope - the slope of the peak exercise ST segment 
#         
#         
#         *0: Upsloping: better heart rate with excercise (uncommon) 
#         *1: Flatsloping: minimal change (typical healthy heart) 
#         *2: Downslopins: signs of unhealthy heart

# ## Experimentation
# 
# At this stage, if you have not gotten the necessary accuracy, you should either addd more data to improve the machine learning or better the current model or try another model like CatBoost and XGBoost.
# 
# If the model is good enough(you have hit the evaluation metric), then you should share it with the client or your friends.
# 

# In[ ]:




