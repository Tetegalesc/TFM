print('=============================================================================')
print('                                   1 APROX')
print('=============================================================================')

# Librerias
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Graphics
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

# Data Loader
from DataLoader import DataLoader

# Model Selector
from ModelSelector import ModelSelector

from sklearn.model_selection import TimeSeriesSplit

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# Hyperparameters Tuning
import hyperopt
from hyperopt.pyll import scope

#######################################################################################################################
# Loading Data
#######################################################################################################################

print('Loading Data...')

df = DataLoader()

print('Data Loaded!!!!')


#######################################################################################################################
# Random State
#######################################################################################################################

seed = 33

#######################################################################################################################
# CV
#######################################################################################################################

splits = 5
temporal_cv = TimeSeriesSplit(n_splits=splits)

#######################################################################################################################
#######################################################################################################################
# Model Selection
#######################################################################################################################
#######################################################################################################################

Selector = ModelSelector(df=df, cv=temporal_cv, seed=seed, metric='balanced_accuracy')

#######################################################################################################################
# Create Label
#######################################################################################################################

Selector.CreateLabel(umb=0.04)

#######################################################################################################################
# Train/Test Split
#######################################################################################################################

Selector.TrainTestSplit()

#######################################################################################################################
# Models Info
#######################################################################################################################

# LR

LR = LogisticRegression()

LR_grid = {'penalty' : ['l2', None],
           'C' : sp_uniform(),
           'solver' : ['lbfgs', 'newton-cg', 'sag', 'saga'],
           'multi_class' : ['ovr'],
           'n_jobs' : [-1],
           'random_state' : [seed]}

# ET

ET = ExtraTreesClassifier()

ET_grid = {'n_estimators' : sp_randint(50, 500),
           'criterion' : ['gini', 'entropy', 'log_loss'],
           'max_depth' : sp_randint(1, 100),
           'min_samples_split' : sp_randint(2, 10),
           'max_features' : ['sqrt', 'log2'],
           'bootstrap' : [False, True],
           'n_jobs' : [-1],
           'random_state' : [seed],
           'class_weight' : ['balanced_subsample']}

# RF

RF = RandomForestClassifier()

RF_grid = {'n_estimators' : sp_randint(50, 500),
           'criterion' : ['gini', 'entropy', 'log_loss'],
           'max_depth' : sp_randint(1, 100),
           'min_samples_split' : sp_randint(2, 10),
           'max_features' : ['sqrt', 'log2'],
           'n_jobs' : [-1],
           'random_state' : [seed],
           'class_weight' : ['balanced_subsample']}

# XGBoost

XGB = XGBClassifier()

XGB_grid = {'n_estimators' : sp_randint(50,250),
            'gamma' : sp_uniform(loc = 0, scale = 11),
            'eta' : sp_uniform(),
            'max_depth' : sp_randint(2,30),
            'seed' : [seed]}

# INFO

names = [
        'LR', 
        'ET',
        'RF',
        'XGB'
        ]

models = [
        LR, 
        ET,
        RF,
        XGB
        ]
grids = [
        LR_grid, 
        ET_grid,
        RF_grid,
        XGB_grid
        ]

#######################################################################################################################
# Selecting Best Model
#######################################################################################################################

Selector.SelectModel(names=names, models=models, models_grid=grids)

#######################################################################################################################
#######################################################################################################################
# Feature Selection
#######################################################################################################################
#######################################################################################################################

Selector.FeatureSelection(model_name='RF')

#######################################################################################################################
#######################################################################################################################
# Hyperparameters Tuning
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Features Selected in Feature Selection
#######################################################################################################################

features_selected = ['close', 'MACD_Hist', 'RSI']

#######################################################################################################################
# Searching Space
#######################################################################################################################

crit = ['gini', 'entropy', 'log_loss']
feat = ['sqrt', 'log2']

space = {'n_estimators' : scope.int(hyperopt.hp.quniform('n_estimators', 50, 200, 1)),
         'criterion' : hyperopt.hp.choice('criterion', crit),
         'max_depth' : scope.int(hyperopt.hp.quniform('max_depth', 2, 30, 1)),
         'min_samples_split' : scope.int(hyperopt.hp.quniform('min_samples_split', 2, 100, 1)),
         'min_samples_leaf' : scope.int(hyperopt.hp.quniform('min_samples_leaf', 1, 50, 1)),
         'min_weight_fraction_leaf' : hyperopt.hp.uniform('min_weight_fraction_leaf', 0, 0.2),
         'max_features' : hyperopt.hp.choice('max_features', feat),
         'n_jobs' : hyperopt.hp.choice('n_jobs', [-1]),
         'random_state' : hyperopt.hp.choice('random_state', [seed]),
         'class_weight' : hyperopt.hp.choice('class_weight', ['balanced_subsample'])}

#######################################################################################################################
# Seeking Best Hyperparameters
#######################################################################################################################

Selector.HyperparametersTuning(model_name='RF', features= features_selected, grid=space)

#######################################################################################################################
#######################################################################################################################
# Final Model
#######################################################################################################################
#######################################################################################################################

features_selected = ['close', 'MACD_Hist', 'RSI']

params = {'class_weight': 'balanced_subsample',
          'criterion': 'gini', 
          'max_depth': 2, 
          'max_features': 'log2', 
          'min_samples_leaf': 18, 
          'min_samples_split': 57, 
          'min_weight_fraction_leaf': 6.4048242025957144e-06, 
          'n_estimators': 181, 
          'n_jobs': -1, 
          'random_state': seed}

final_model = RandomForestClassifier(**params)

Selector.FinalModel(model_name='RF', model=final_model, features=features_selected)