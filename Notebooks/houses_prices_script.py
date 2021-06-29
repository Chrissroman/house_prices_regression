# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Charge of libaries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from scipy import stats
from scipy.stats import norm, skew
from math import ceil


color = sns.color_palette()
sns.set_style('darkgrid')

# Path directory when is all files of the study
os.chdir("R:\Respaldo\Christian\Kaggle\Houses Prices")
os.getcwd()

# Charge data in DataFrame
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# Save ID columns
id_train = train['Id']
id_test = test['Id']
id_submission = submission['Id']

train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)
submission.drop('Id', axis = 1, inplace = True)

print("The size of train dataset is {} and the test is {}"
      .format(train.shape, test.shape))

####################### DATA PROCESSING AND ANALYSIS ########################

"""                         OUTLIERS

Let's explore these ouliers in principal variables float's

"""
fig, ax = plt.subplots(figsize = (16,9))
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('SalePrice', fontsize=20)
plt.ylabel('SalePrice', fontsize=20)
plt.show()

"""
We can see at the bottom right two elements with extremely large of GrLivArea 
that are of a low price. These values are outliers clearly.  
Therefore, we can proceed to delete them.
"""

train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index,
                   axis=0)

# Check the graphic again
fig, ax = plt.subplots(figsize = (16,9))
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('GrLivArea', fontsize = 20)
plt.ylabel('SalePrice', fontsize = 20)
plt.show()

"""                         NORMALIZE TARGET VARIABLE

We can need to do some analysis on this SalePrice variable. These is variable predict

"""

# Target Variable Analysis
sns.distplot(train['SalePrice'], fit=norm)

# Get the fitted parameters used by the fuction
(mu, sigma) = norm.fit(train['SalePrice'])
print("The Parameters mu = {:.2f} and sigma = {:.2f}".format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],
           loc = 'best')

plt.ylabel("Frecuency")
plt.title("SalePrice Distribution")

# Get also a the Probality Plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
plt.show()


"""
with the purpose of obtaining to the linear model, 
we need to transform this variable at a normal distribution.
"""

train['SalePrice'] = np.log1p(train['SalePrice'])

sns.distplot(train['SalePrice'], fit=norm)

# Get the fitted parameters used by the fuction
(mu, sigma) = norm.fit(train['SalePrice'])
print("The Parameters mu = {:.2f} and sigma = {:.2f}".format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],
           loc = 'best')

plt.ylabel("Frecuency")
plt.title("SalePrice Distribution")

# Get also a the Probality Plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
plt.show()

"""
Now the data appears more normally distributed. Also, we did to corrected the skew"""

"""                            FEATURES ENGINEERING
"""

# Concatenate the train and test data in the same DataFrame

# We save the size data
n_train = train.shape[0]
n_test = test.shape[0]
n_data = n_train + n_test

# Save target Variable of Train Data
y_target = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop = True)
all_data.drop(['SalePrice'], axis = 1, inplace = True)
print("The data sets has size of: {}".format(all_data.shape))


"""                              MISSING VALUES         

We does manage of data that contain missing values,
somes will can drop and others does replace with a calculate. 
This depend of type data, its importance and amount missing values

"""

def miss_values(df):
    
    columns = df.columns.values.tolist()
    cols = []
    
    for c in columns:
        miss = pd.isnull(df[c]).sum()
        if miss >0:
            cols.append([c, round(miss/len(df)*100, 2), miss, df[c].dtypes])
    
    isnull = pd.DataFrame(cols).rename({0: 'Feature',
                                        1: '% Null',
                                        2: 'Nº',
                                        3: 'dtype'},
                                       axis = 1)
    
    isnull = isnull.set_index(['Feature'])\
        .sort_values(['% Null'], ascending = False)
        
    return isnull

isnull_before = miss_values(all_data)
print("The feutures had missing values or NA is: ")
isnull_before.head(len(isnull_before))

# We do graphic a the bar plot those features that have missing values
fig, ax = plt.subplots(figsize=(16,9))
plt.xticks(rotation='90', fontsize = 15)
plt.yticks(fontsize = 15)
sns.barplot(x = isnull_before.index, y = '% Null', palette = 'Blues_d', 
            data = isnull_before)
plt.xlabel('Features', fontsize = 18)
plt.ylabel('% Null', fontsize = 18)
plt.title("Percent of missing values by feature", fontsize = 20)

# could have importance that we see the correlations level eache feature
corr_all = train.corr(method = 'pearson')

# We Plot a Heatmap to see this level correlation intuitively

mask = np.zeros_like(corr_all)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize = (32, 18))
    fig.suptitle('Matrix of correlations', fontsize = 20)
    ax = sns.heatmap(corr_all, linewidths=0.1, vmin = -0.5, annot = True,
                     annot_kws= {'size' :10}, fmt = '.1g' ,vmax = 0.9, square = True, 
                     mask = mask, cmap = "vlag")

"""                          IMPUTING MISSING VALUES                       """

""" 
    Some features have missing values are discreet and others numerics. 
We build two iteration loops for fill missing values or NA with respective values 'None' or '0'

The others features require personalizing treatment, for example, LotFrontage and KitchenQual.

"""
#                              DISCRETE VARIABLES

# PoolQC: PoolQC has NA values when these discribe "No Poll". This explain as had 99% of missing values
# MiscFeature: the data description that mark NA as None Miscellaneous Features
# Alley: the data description that mark NA as None Alley access street
# Fence: the data description points out that the NA value as No Fence
# FireplaceQu: the data description points out that the NA value as No Fireplace when Fireplace is zero
# Garage: the features described from of 'Garage' features, points out that the NA values is No Garage
# Basement: the features described from of 'basement' features, point out that the NA values is No Basement
# MasVnrType: for options in this feature is None (without type masonary veneer). 
#       Therefore, we fill missing values with 'None'.


cols_fill = ['PoolQC','MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish',
             'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']

for c in cols_fill:
    all_data[c].fillna('None', inplace = True)
    
    

# LotFrontage: as it is a Linear Feet Squared of Street that connects to the property, 
# we can build a median where it will depend on the neighborhood. With this metric,
# we inputing missing values in this feature.
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# MSZoning: this feature describes the Classification Zoning of house, 
#           for example, 'Agriculture, Comercial, Residential, industrial, between others.
#           We located the higher frequency of this classification into the Neighborhood and fill these missing values with this.
MSzone_gb = all_data.groupby('Neighborhood')['MSZoning'].value_counts()

for i in range(0, len(all_data)):
          
    if pd.isnull(all_data['MSZoning'].iloc[i]) == True:
        
       nbh = MSzone_gb[all_data['Neighborhood'].iloc[i]].index[0][1]
        
       all_data['MSZoning'].iloc[i] = nbh    

# Functional: This feature describe the home functional. The higher frecuency found is
#           'Typ' (Typical functionality). We fill NaN values with this data.

all_data['Functional'].value_counts()

all_data['Functional'].fillna('Typ', inplace = True)

# Utilities: This feature describes the general home utilities. The higher frequency found is
#           'AllPub' (All Public Utilities) and are these the only values except one that is NoSewr.
#           We believe it convenient to drop these columns because they same not offer value at the model.
all_data['Utilities'].value_counts()

idx = all_data[all_data['Utilities'] == 'NoSeWa'].index.values
print(idx) # -->  the only value is not (all public utilities is train data set)
all_data.drop('Utilities', axis = 1, inplace = True) 


# Electrical: This feature describe the electrical conditions. The higher frecuency found is
#           'SBrkr' (Standard Circuit Breakers & Romex). We fill NaN values with this data.

all_data['Electrical'].value_counts()
all_data['Electrical'].fillna('SBrkr', inplace = True)

# KitchenQual: This feature describe the kitchen quality. The higher frecuency found is
#           between 'TA' (Typical/Average) and 'Gd' (Good). We fill NaN values with this data.
KitchenQ_gb = all_data.groupby('Neighborhood')['KitchenQual'].value_counts()
idx = pd.isnull(all_data['KitchenQual'])
idx = idx[idx == True].index.values

fill_value = KitchenQ_gb[all_data['Neighborhood'].iloc[idx]].index[0][1]
all_data['KitchenQual'].iloc[idx] = fill_value

# Exterior1st: This feature describe the first material exterior as was bluid. The higher frecuency found is 
#            'VinylSd' (Vinyl Siding). We fill NaN values with this data.
all_data['Exterior1st'].value_counts()
all_data['Exterior1st'].fillna('VinylSd', inplace = True)

# Exterior2nd: This feature describe the second material exterior as was bluid. The higher frecuency found is 
#            'VinylSd' (Vinyl Siding). We fill NaN values with this data.
all_data['Exterior2nd'].value_counts()
all_data['Exterior2nd'].fillna('VinylSd', inplace = True)

# SaleType: This feature describe Sale Type. Intuitively we can see one correlation between
#           Sale Condition and Sale Type. With this table grouped of these two variables, 
#           is can describe its influence clearly
SaleType_gb = all_data.groupby('SaleCondition')['SaleType'].value_counts()
idx = pd.isnull(all_data['SaleType'])
idx = idx[idx == True].index.values

fill_value = SaleType_gb[all_data['SaleCondition'].iloc[idx]].index[0][1]
all_data['SaleType'].fillna(fill_value, inplace = True)

#                               CONTINUUM VARIABLES

# GarageYrBlt: This feature describe is year garage was build. The NaN values coincide with
#           'None' values in Garage Type (No Garage). Thus, we inputing zero value.
# GarageArea, GarageCars: This feature is same conditions and after variable describe.
# MasVnrArea: This feature describe the square feet of masonry veneer. the NaN values coincide with 
#           'None' values in Masonry Veneer Type. Thus, we fill missing values with zero. 
# BsmtUnfSF, BsmtFinSF1, BsmtFinSF2, TotalBsmtSF: Is features what is desccribe of square feet
#           these variables are part of basement conditions both completed, unfinished as No Basement

cols_fill = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea',
             'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF',
             'BsmtFullBath', 'BsmtHalfBath']

for col in cols_fill:
    
    all_data[col].fillna(0, inplace = True)

""" We can see the status of missing values after treatment variables. """
isnull_after = np.sum((all_data.isnull().sum()/len(all_data))) *100
print("The Percentage of missing values is: {:.2f}%".format(isnull_after))

"""                         ENGENEERING OF VARIABLES

Some feauture require to addition tratment. Variebles as OverallCond is cateogrical 
character but, it is type as float64. We converse these variables in type string. 

"""

# Feature at convert:
    # transformation in string
cols = ['OverallCond', 'OverallQual', 'YrSold', 'MoSold', 'MSZoning']
all_data[cols] = all_data[cols].astype(str, copy = True)
    
    # transformation in intenger
cols = ['BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageYrBlt', 'YearBuilt',
        ]
all_data[cols] = all_data[cols].astype(int, copy = True)

    # transformation in float64
cols = ['1stFlrSF', '1stFlrSF', 'LowQualFinSF', 'GrLivArea', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
        'LotArea', ]
all_data[cols] = all_data[cols].astype(float, copy = True)


all_data.info()

#                             ENCODING VARIABLLES

""" 
We have some tools for the encoding of categorical variables. In this DataSet,
we have many categorical features with multiplex options whose ordering
has much information thus, dummy variables aren't options. One elegant solution 
could be to encoding variables in progression numeric from 0 to n-options. 
Scikit-Learn have tools for doing to encode:
    
For Algorithms of Machine Learning, manage numeric data type is most efficient. 

"""

# Selection features for encoding
cols_encoder = ['MSSubClass', 'LotShape', 'Alley', 'OverallQual', 'OverallCond',
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
        'Fence', 'MoSold']

# Import library of sklearn for the econding varibles:
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()

for col in cols_encoder:
    all_data[col] = lbe.fit_transform(list(all_data[col].values))
    
all_data.info()
    
#                             TRANSFORMATION

""" 
The must feature have a bias higher at the absolute value of | 1 |. A good practice
could be to transform each variable with a suitable function.

The Box-Cox transformation is a good option for multiple available transformations to have.
We iterate a Lambda value until to getting a transformation that comes closer to an
normal distribution.

"""

features = all_data.dtypes[all_data.dtypes == 'float'].index

# we extract the bias value and order.
skewed_feats = all_data[features].apply(lambda x: skew(x))\
    .sort_values(ascending = False)
print(skewed_feats)

# We select those variables that have bias higher than 1
feats_bias = skewed_feats[abs(skewed_feats) >= 1].index
print('Features selection: \n', feats_bias)


# can be convenient that we see the distribution of these features.
n_cols = 3
n_rows = ceil(len(feats_bias)/n_cols)

fig, ax = plt.subplots(ncols = n_cols , nrows = n_rows, figsize=(16,16))

for i, col in enumerate(all_data[feats_bias]):
    
    axes = ax[i//n_cols, i%n_cols]
    sns.distplot(all_data[col], fit = norm, kde = False, 
                 ax = axes)
    axes.legend(['bias: {:.2f}'.format(skew(all_data[col]))], 
               loc = 'best')
    fig.suptitle(t = 'Features that are skewed with a value higher than |1.|', 
                 fontsize = 20)
    plt.tight_layout()


######################## TRANSFORMATION BOX COX (1 + X) ######################

""" 
We build two simple functions that search at optimal lambda that to reduce
of the standard deviation values close to zero.
"""

def skew_asymmetry(X, th_skew = (-1, 1), th_cv = 0.8):
    
    # if CV> 0.8, the destribution data is considered homogeneous
    # if CV <= 0.8, the distribution data is considered heterogeneous  
    # th_skew = Threshold considered for we catalog variables as asymmetrical.
    
    # library import 
    from scipy.stats import skew
    
    asymmetry = skew(X)
    
    if (th_skew[0] > asymmetry > th_skew[1]): #or cv > th_cv:
        return True
    else:
        return False

def search_lambda_boxcox1p(df):
    
    #library import
    import numpy as np
    import pandas as pd
    from scipy.stats import boxcox
    # Range lambada for iterate    

    list_feat = []
    
    for feat in df.columns.tolist():      
        
        if skew_asymmetry(df[feat]) == False:
            
            _, lm = boxcox(df[feat] + 1) #
            list_feat.append([feat, lm])
            
        
        array_feat = np.array(list_feat)
        df_lm = pd.DataFrame(data = array_feat, columns = ['feature', 'opt-lambda'])
        df_lm.set_index('feature', inplace=True)
            
    return df_lm 

"""
TRANSFORM BOX-COX METHOD
"""
##################
from scipy.special import boxcox1p

# Selection of lambda parameter

all_data_t = all_data.copy()
df_lambda = search_lambda_boxcox1p(all_data_t[feats_bias])

# Transform each variable with box-cox (1+x) with boxcox1p from scipy
for col in feats_bias:
    #↓all_data_t[col] = boxcox1p(all_data_t[col].to_numpy(), 0.25)
    all_data_t[col] = boxcox1p(all_data_t[col].to_numpy(), float(df_lambda.loc[col][0]))

# Plot the new distribution 
n_cols = 3
n_rows = ceil(len(feats_bias)/n_cols)

fig, ax = plt.subplots(ncols = n_cols , nrows = n_rows, figsize=(16,16))

for i, col in enumerate(all_data_t[feats_bias]):
    
    axes = ax[i//n_cols, i%n_cols]
    sns.distplot(all_data_t[col], fit = norm, kde = False, 
                 ax = axes)
    axes.legend(['bias: {:.2f}'.format(skew(all_data_t[col]))], 
               loc = 'best')
    fig.suptitle(t = 'New distribution with Box-Cox transform method', 
                 fontsize = 20)
    plt.tight_layout()
    
#######################

#                               DUMMY VARIABLES

"""                            
    An idea widely used is to create dummy variables from categorical features. 
This is very efficient because, the model of machine learning return the biggest 
precision with binary features (1 or 0). 
"""

# Drop the old features that was transform in dummy variables
all_data_t = pd.get_dummies(all_data_t, prefix = None, prefix_sep = '_', 
                            drop_first = True)

all_data_t.shape

data_train = all_data_t[:n_train]
data_test = all_data_t[n_test - 1:]
print("Size of the data train is {}, and the data test: {}".format(data_train.shape, 
                                                                   data_test.shape))







##############################################################################
# ============================================================================
#                                MODELING
# ============================================================================

"""
    We have many models and algorithms that have objective predict a quantitive value, 
in this case, Sales Price of Houses as target feature. XGBoost, Gradient Boosting Regression
and Random Forest Regression (Also the Traditional Desisión Forest) 
are options more than enough for modelling the behaviour of our data set.

In this case, we can evaluate some algorithms that allow us to do cross-validation 
between different models. We start with a Regression Tree with cross-validation 
after the random forest and finaly XGBoost that is the algorithms recommended 
for the competition.

""" 

# Import libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error
import joblib
import multiprocessing

df_submission = pd.DataFrame()
df_submission = np.log1p(submission)

X_train, X_test, y_train, y_test = train_test_split(data_train, y_target,
                                                    test_size = 0.3,
                                                    random_state = 2021)
    
#                         RANDOM FOREST REGRESSION

# Load Model was training 
forest = joblib.load('random_forest_model.sav')

forest = RandomForestRegressor(n_jobs = multiprocessing.cpu_count() - 1, 
                               oob_score= True, n_estimators = 3500,
                               criterion = 'mse', max_depth = 6, 
                               max_features = 'auto') # -> Mean Squared Error as Support Criteria

# Training the Random Forest
forest.fit(X_train, y_train)

df_regression = pd.DataFrame()
df_regression['y_pred'] = y_test
df_regression['rforest_pred'] = forest.predict(X_test)

df_regression.head()

df_regression['rforest_error2'] = (df_regression['rforest_pred'] - df_regression['y_pred']) **2
rforest_error2 = sum(df_regression['rforest_error2'])/len(data_train)
print("The Median Squared Error in Random Forest Regression: {:.4f}".format(rforest_error2))
print("The Coefficient of Determination for Random Forest Regression Mothel is: {:.4f}".format(forest.oob_score_))


# Save prediction on Data Testing in DataFrame
df_submission['random_forest'] = forest.predict(data_test)

# Mean Squared Error of Data Testing prediction.
print("The mean squared error of model between submission and predict:")
print(mean_squared_error(df_submission['SalePrice'], df_submission['random_forest']))

# The list of features importances for in Random Forest model.
forest_feats = pd.DataFrame(forest.feature_importances_, 
                            index = data_train.columns.tolist(),
                            columns = ['feature_importances'])

forest_feats.sort_values(by = 'feature_importances', ascending = False, inplace = True)

# The frist ten features importance in the model.
forest_feats.head(20)

#======= SAVE MODEL =======
filename = 'random_forest_model.sav'
joblib.dump(forest, filename)
# =========================






# ============================================================================
#                       GRADIENT BOOSTING REGRESSION


###############################################################################
#                       N° OF TREE - CROOS VALIDATION

# Validaton with k-cross_validation and neg_root_mean_squared_error
# ============================================================================
train_scores = []
cv_scores = []

# Values at evalueate
estimator_range = range(1, 10000, 500)

# Loop for training model for each value of n_estimators, 
# after we extract its error of train and k-cross-validation.
for n_estimators in estimator_range:
    
    GBR_model = GradientBoostingRegressor(loss = 'ls',
                                          learning_rate     = 0.01,
                                          n_estimators      = n_estimators,
                                          criterion         ='mse',
                                          min_samples_split = 15,
                                          min_samples_leaf  = 10,
                                          max_depth         = 4,
                                          max_features      = 'sqrt',
                                          #--> Range in percentage wherefore we consider the model is not getting better.
                                          tol               = 0.0001,  
                                          random_state      = 2021
                                          )
    # Error of train
    GBR_model.fit(X_train, y_train) #--> Train
    GBR_pred = GBR_model.predict(X_test) #--> Pred Values
    GBR_rsme = mean_squared_error(y_true  = y_test, #--> Mean Squared Errot
                                  y_pred  = GBR_pred,
                                  squared = False)
    
    train_scores.append(GBR_rsme)
    
    # Error Cross Validation
    cv = KFold(n_splits = 10, shuffle=True, random_state = 2021)
    GBR_scores = cross_val_score(
                    estimator = GBR_model,
                    X         = X_train,
                    y         = y_train,
                    scoring   = 'neg_root_mean_squared_error',
                    cv        = cv,
                    n_jobs    = multiprocessing.cpu_count() - 1              
                    )
    
    # We add the scores of cross_val_score() and covert positive
    cv_scores.append(-1*GBR_scores.mean())
    
# Plot with the evolution of errors:
fig, ax = plt.subplots(figsize = (12, 8))
ax.plot(estimator_range, train_scores, label = 'Train Scores')
ax.plot(estimator_range, cv_scores, label = "Cross-Validation Scores")
ax.plot(estimator_range[np.argmin(cv_scores)], min(cv_scores),
        marker = 'o', color = 'red', label = 'Min Score')
ax.set_xlabel("n_estimators")
ax.set_ylabel("root_mean_squared_error")
ax.set_title("The evolution CV Error vs N° of Estimators")
plt.grid()
plt.legend()

print("The N° Estimators optimal {}, where is reached a CV Min Error of {:.4f}".format(estimator_range[np.argmin(cv_scores)], min(cv_scores)))


print("The Error (RMSE) of test in Gradient Boosting Regressor model is: {:.4f}".format(GBR_rsme))

##############################################################################
#                   LEARNING RATE  N° of TREE - CROSS VALIDATION

# Validaton with k-cross_validation and neg_root_mean_squared_error
# ============================================================================
results = {}

# Learning rate values and n_estimators
learning_rates = [0.001, 0.01, 0.1]
n_estimators = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 8000, 10000]

# Loop for training a model for each combination of learning rate and n_estimatior
# in addition, we extract of Mean Squared Error and K-Cross-Validation

for lr in learning_rates:
    
    train_scores = []
    cv_scores = []
    
    for n_trees in n_estimators:
        
        # Set paramns model
        GBR_model = GradientBoostingRegressor(
                            loss                = 'ls',
                            learning_rate       = lr,
                            n_estimators        = n_trees,
                            criterion           = 'mse',
                            min_samples_split   = 15,
                            min_samples_leaf    = 10,
                            max_depth           = 4,
                            max_features        = 'sqrt',
                            #--> Range in percentage wherefore we consider the model is not getting better.
                            tol                 = 0.0001,
                            random_state        = 2021 
            )

        # Error of train
        GBR_model.fit(X_train, y_train)
        GBR_pred = GBR_model.predict(X_test)
        GBR_rmse = mean_squared_error(y_true   = y_test,
                                      y_pred   = GBR_pred,
                                      squared  = False
                                      )
        train_scores.append(GBR_rmse)
        
        # Error Cross - Validation
        cv = KFold(n_splits = 10, shuffle=True, random_state = 2021)
        GBR_scores = cross_val_score(
                                estimator       = GBR_model, 
                                X               = X_train, 
                                y               = y_train,
                                scoring         = 'neg_root_mean_squared_error',
                                cv              = cv,
                                n_jobs          = multiprocessing.cpu_count() - 1
                                ) 

        # We add the cross_val_score and convert positive.
        cv_scores.append(-1*GBR_scores.mean())
        
    results[lr] = {'train_scores': train_scores, 'cv_scores': cv_scores} 

# Plot with evaluation of errors:
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 8))

for key, value in results.items():
    axes[0].plot(n_estimators, value['train_scores'], label = "Learning rate {}".format(key))
    axes[0].set_xlabel("n_estimators")
    axes[0].set_ylabel("root_mean_squared_error")
    axes[0].set_title("The Evolution Train Error vs Learning Rate")
    
    axes[1].plot(n_estimators, value['cv_scores'], label = "learning rate {}".format(key))
    axes[1].set_xlabel("n_estimators")
    axes[1].set_ylabel("roor_mean_squared_error")
    axes[1].set_title("The Evolution CV Error vs Leraning Rate")
    plt.grid()
    plt.legend()


#############################################################################
#                   MAX DEPTH - CROSS - VALIDATION

# Validation with k-cross-validation and neg_root_mean_squared_error
# ===========================================================================

train_scores = []
cv_scores = []

# Values at Evaluate
max_depths = [3, 4, 5, 6, 7, 8, 9, 10, 20]

for depth in max_depths:
    
    GBR_model = GradientBoostingRegressor(
                        loss = 'ls',
                        learning_rate     = 0.11,
                        n_estimators      = 2500,
                        min_samples_split = 15,
                        min_samples_leaf  = 10,
                        max_depth         = depth,
                        max_features      = 'sqrt',
                        #--> Range in percentage wherefore we consider the model is not getting better.
                        tol             = 0.0001,
                        random_state      = 2021
        )

    # Error of Train
    GBR_model.fit(X_train, y_train)
    GBR_pred = GBR_model.predict(X_test)
    GBR_rmse = mean_squared_error(y_true  = y_test,
                                  y_pred  = GBR_pred,
                                  squared = False
                                  )
    train_scores.append(GBR_rmse)
    
    # Error Cross - Validation
    cv = KFold(n_splits = 10, shuffle = True, random_state = 2021)
    GBR_scores = cross_val_score(estimator = GBR_model,
                                 X         = X_train,
                                 y         = y_train,
                                 scoring   = 'neg_root_mean_squared_error',
                                 cv        = cv,
                                 n_jobs    = multiprocessing.cpu_count() - 1
                                 )
    
    # We add the cross_val_score and convert positive
    cv_scores.append(-1*GBR_scores.mean())

fig, ax = plt.subplots(figsize = (12, 8))
ax.plot(max_depths, train_scores, label = "Train Scores")
ax.plot(max_depths, cv_scores, label = "CV Scores")
ax.plot(max_depths[np.argmin(cv_scores)], min(cv_scores), label = "Min Score",
        marker = "o", color = 'red')
ax.set_xlabel("max_depth")
ax.set_ylabel("root_mean_squared_error")
ax.set_title("Th Evolution CV Error vs the Max Depth of Treee")
plt.grid()
plt.legend()

print("The optimal value for Max Depth is: {}".format(max_depths[np.argmin(cv_scores)]))



#############################################################################
#                  TUNING GBR MODEL USING GRID SEARCH

"""
The other form of search the best hyperparameters that are cotained in 
Gradiente Boosting Regressor Algorithm is pass a grid paramns for the search.
The Grid Search will find the best combination that keeps equilibrium 
between the bias and variance.

"""

# Grid of Hyperparameters Evaluated
# ===========================================================================

# Load model of grid search for Gradient Boosting Regressor
grid = joblib.load("GBR_grid_search.sav")

param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': [2, 4, 6, 10],
              'subsample': [0.5, 1],
              'learning_rate': [0.001, 0.01, 0.1],
              'min_samples_split': [14, 16, 18],
              'min_samples_leaf': [6, 10, 12]
    }

# We pass of grid parameters. We keep the type scoring to add an KFold Sthocastic
# with RepeatedKFold from Scikit-Learn.

# Grid Search of hyperparameters
grid = GridSearchCV(
    estimator           = GradientBoostingRegressor(
                                loss                = 'ls',
                                n_estimators        = 2500,
                                random_state        = 2021,
                                # For Early
                                validation_fraction = 0.1,
                                tol                 = 0.0001
                                ),
    param_grid         = param_grid,
    scoring            = 'neg_root_mean_squared_error',
    n_jobs             = multiprocessing.cpu_count() -1,
    cv                 = RepeatedKFold(n_splits     = 10,
                                       n_repeats    = 1,
                                       random_state = 2021),
    refit              = True,
    verbose            = 0, 
    return_train_score = True
    )

# Fit Grid Search.
grid.fit(X_train, y_train)

GBR_model_gs = pd.DataFrame(grid.cv_results_)

GBR_model_gs.filter(regex = '(param.*|mean_t|std)') \
    .drop(columns = 'params') \
        .sort_values('mean_test_score', ascending = False) \
            .head(8)

# ==== SAVE GRID ====
filename = 'GBR_grid_search.sav'
joblib.dump(grid, filename)
# ===================

# ===========================================================================
# The best hyperparamns finded:
print("---------------------------------------------------------------------")
print("The best Hyperparamns finded (CV)")
print("---------------------------------------------------------------------")
print(grid.best_params_, ":", grid.best_score_, grid.scoring)    


"""
----> ---------------------------------------------------------------------
      The best Hyperparamns finded:
      ---------------------------------------------------------------------
      {'learning_rate': 0.01, 'max_depth': 6, 'max_features': 'sqrt', 
       'min_samples_leaf': 6, 'min_samples_split': 18, 'subsample': 0.5} : 
          -0.11671019046652402 neg_root_mean_squared_error
"""

#############################################################################
#                 MODEL OPTIMAL FOR GRADIENT BOOSTING REGRESSOR

"""
One time we find th hyperparametrs optimal, we built the definitive model with its.

"""
GBR_model_def = joblib.load("GBR_model.sav")

GBR_model_def = GradientBoostingRegressor(
                                loss              = 'ls',
                                learning_rate     = 0.01,
                                n_estimators      = 2500,
                                max_features      ='sqrt',
                                min_samples_split = 18,
                                min_samples_leaf  = 6,
                                max_depth         = 6,
                                subsample         = 0.5,
                                # For Early
                                validation_fraction = 0.1,
                                tol = 0.0001
                                )

GBR_model_def.fit(X_train, y_train)
GBR_pred = GBR_model_def.predict(X_test)
GBR_rmse = mean_squared_error(y_true = y_test,
                              y_pred = GBR_pred)

print(" The (RMSE) in Final Model of Gradient Boosting Regressor is: {:.4}".format(GBR_rmse))
print("The Coefficient of Determination for Random Forest Regression Mothel is: {:.4f}".format(GBR_model_def.score(X_test, y_test)))

df_regression['GBR_pred'] = GBR_pred
df_regression['GBR_error2'] = (df_regression['GBR_pred'] - df_regression['y_pred'])**2

# Save Prediction of Data Testing on DataFrame 
df_submission['GBR'] = GBR_model_def.predict(data_test)

# Mean Squared Error of Data Testing prediction.
print("The mean squared error of model between submission and predict:")
print(mean_squared_error(df_submission['SalePrice'], df_submission['GBR']))

# ==== SAVE MODEL ====
filename = 'GBR_model.sav'
joblib.dump(GBR_model_def, filename)
# ====================





# ============================================================================
#                 LASSO - LINEAR MODEL REGRESSION WITH PENELTY


"""
    Linear Regression Ordinary have a disadvantage clearly, the model take all
features with the same importance and cannot penalty those features with little
 influence as well the random forest model and gradient boosting regression 
 do yet.

LASSO model is a Linear Regression with Lambda Factor that allows to penalty 
those features not offer importance in the model. Is evident that a 
Cross-Validation is necessary as an iterative method for finding this optimal 
lambda.

Scikit-Learn have a function LassoCV that have inconporated the cross-validation
for the alpha (lambda) parameter. But, if we wanna to obtain a tunning model, using Grid Search
should return all ideals hyperparameters. 
"""

##############################################################################
#                       TUNING LASSO MODEL USING GRID SEARCH

param_grid = {'alpha': np.arange(0, 1, 0.1),
               'fit_intercept': [True, False],
               'selection': ['cyclic', 'random'],
    }

cv = RepeatedKFold(n_splits      = 10,
                   n_repeats    = 1,
                   random_state  = 2021)

grid_search = GridSearchCV(
                estimator = Lasso(random_state = 2021,
                                  tol          = 0.001),
                param_grid         = param_grid,
                cv                 = cv,
                scoring            = 'neg_root_mean_squared_error',
                n_jobs             = multiprocessing.cpu_count()-1,
                refit              = True,
                verbose            = 0,
                return_train_score = True             
    )

grid_search.fit(X_train, y_train)

LASSO_model_gs = pd.DataFrame(grid_search.cv_results_)

LASSO_model_gs.filter(regex = '(param.*|mean_t|std)') \
    .drop(columns = 'params') \
        .sort_values('mean_test_score', ascending = False) \
            .head(8)


# ===========================================================================
# The best hyperparamns finded:
print("---------------------------------------------------------------------")
print("The best Hyperparamns finded (CV)")
print("---------------------------------------------------------------------")
print(grid_search.best_params_, ":", grid_search.best_score_, grid_search.scoring)    


###### SAVE MODEL #####
joblib.dump(grid_search, "LASO_grid_search.sav")
######################

##############################################################################
#          MODEL OPTIMAL FOR LASSO REGRESSOR (Clasic Linear Regressor)

# We build the definitve model with the best alpha that return the Mena Test
# Score 

LASSO_model = Lasso(alpha = 0.0005, random_state = 2021)

LASSO_model.fit(X_train, y_train)
LASSO_pred = LASSO_model.predict(X_test)

score = LASSO_model.score(X_test, y_test)

print("The Coefficient of Determination of prediction is {:.4f}".format(score))

# Save prediction of X_test in DataFrame Regression
df_regression['LASSO_pred'] = LASSO_pred
df_regression['LASSO_error2'] = (df_regression['LASSO_pred']- df_regression['y_pred']) **2

# Save prediction of Data Testing in DataFrame Submission
df_submission['LASSO'] = LASSO_model.predict(data_test) 

# Mean Squared Error of Data Testing prediction.
print("The mean squared error of model between submission and predict:")
print(mean_squared_error(df_submission['SalePrice'], df_submission['LASSO']))


############# SAVE MODEL #############
filename = 'LASSO_model.sav'
joblib.dump(LASSO_model, filename)
#####################################

##############################################################################
#                               XGBOOST

import xgboost as xgb
print(xgb.__version__)


model = xgb.XGBRegressor(n_estimators     = 2500,
                         max_depth        = 4,
                         eta              = 0.01,
                         subsample        = 0.7,
                         colsample_bytree = 0.5)

cv  = RepeatedKFold(n_splits = 10, n_repeats=1, random_state = 2021)

scores = cross_val_score(model, X_train, y_train, scoring = 'neg_mean_squared_error',
                         cv = cv, n_jobs = multiprocessing.cpu_count() -1)

scores = np.absolute(scores)













































































