print('=============================================================================')
print('                                   2 APROX')
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

Selector.FeatureSelection(model_name='ET')

#######################################################################################################################
#######################################################################################################################
# Hyperparameters Tuning
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Features Selected in Feature Selection
#######################################################################################################################

features_selected = ['open', 'close', 'SMA5', 'SMA15', 'MACD', 'MACD_Hist']

#######################################################################################################################
# Searching Space
#######################################################################################################################

crit = ['gini', 'entropy', 'log_loss']
feat = ['sqrt', 'log2']

space = {'n_estimators' : scope.int(hyperopt.hp.quniform('n_estimators', 100, 500, 1)),
         'criterion' : hyperopt.hp.choice('criterion', crit),
         'max_depth' : scope.int(hyperopt.hp.quniform('max_depth', 1, 15, 1)),
         'min_samples_split' : scope.int(hyperopt.hp.quniform('min_samples_split', 2, 100, 1)),
         'max_features' : hyperopt.hp.choice('max_features', feat),
         'n_jobs' : hyperopt.hp.choice('n_jobs', [-1]),
         'random_state' : hyperopt.hp.choice('random_state', [seed]),
         'class_weight' : hyperopt.hp.choice('class_weight', ['balanced_subsample'])}

#######################################################################################################################
# Seeking Best Hyperparameters
#######################################################################################################################

best_hyper = Selector.HyperparametersTuning(model_name='ET', features= features_selected, grid=space)

#######################################################################################################################
#######################################################################################################################
# Final Model
#######################################################################################################################
#######################################################################################################################

features_selected = ['open', 'close', 'SMA5', 'SMA15', 'MACD', 'MACD_Hist']

params = {'class_weight': 'balanced_subsample',
          'criterion': 'entropy',
          'max_depth': 13,
          'max_features': 'sqrt',
          'min_samples_split': 13,
          'n_estimators': 425,
          'n_jobs': -1,
          'random_state': seed}

final_model = ExtraTreesClassifier(**params)

Selector.FinalModel(model_name='ET', model=final_model, features=features_selected)