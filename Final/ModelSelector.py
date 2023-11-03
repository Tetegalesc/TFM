import os
import pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Model Selection
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
# Feature Selection
from sklearn.inspection import permutation_importance
# Hyper Tuning
import hyperopt
from hyperopt.pyll import scope
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# Metrics
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score, precision_score, f1_score, confusion_matrix, classification_report



class ModelSelector():

    def __init__(self, df, cv, seed, metric):
        # DF
        self.df = df
        # CV
        self.cv = cv
        # Random state
        self.seed = seed

        # Metric
        self.metric = metric

        #TRAIN/TEST SPLIT
        self.X = None
        self.X_train = None
        self.X_train_val = None
        self.X_test = None

        self.y = None
        self.y_train = None
        self.y_train_val = None
        self.y_test = None

        # FEATURES SELECTION
        self.best_feat = None

        # HYPERPARAMETERS TUNING
        self.hyper_space = None
        self.best_hyper = None
        self.best_hyper_score = None

        # FINAL MODEL
        self.y_pred = None

        # RESULTS
        self.confusion_matrix = None
        self.class_rep = None
        self.accuracy = None
        self.recall = None
        self.bal_accuracy = None
        self.precision = None
        self.f1 = None

    def CreateLabel(self, umb=0.04):
        def Label(gain):
            if gain < -umb:
                return 0
            elif -umb <= gain <= umb:
                return 1
            else:
                return 2

        # Create Lable column
        self.df['LABEL'] = self.df['gain'].apply(Label)

        # Save Files
        Data_results_path = 'Results/0_Data'
        os.makedirs(Data_results_path, exist_ok=True)

        # Plot Gain and umb
        plt.plot(self.df['date'], self.df['gain'])
        plt.axhline(y=umb, color='g')
        plt.axhline(y=-umb, color='r')

        plt.title('Umbral Selected')
        plt.xlabel('Date')
        plt.ylabel('Gain')

        plt.savefig('Results/0_Data/Gain_umb.png')
        plt.close()

        # Plot Label Results
        plt.hist(self.df['LABEL'])
        plt.title('Labels')

        plt.savefig('Results/0_Data/Label.png')
        plt.close()

        # Delete 'gain' column
        self.df.drop('gain', axis=1, inplace=True)

    def TrainTestSplit(self):
        # Split data and label
        self.X = self.df.drop('LABEL', axis=1)
        self.y = self.df['LABEL']

        # Train/Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=False)

        # Train_val/Val split
        self.X_train_val, self.X_val, self.y_train_val, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, shuffle=False)

        # Plot Train/Test split
        plt.plot(self.X_train['date'], self.X_train['close'], label='Train')
        plt.plot(self.X_test['date'], self.X_test['close'], label='Test')
        plt.plot(self.X_val['date'], self.X_val['close'], label='Val')

        plt.xlabel('Date')
        plt.ylabel('Close')

        plt.legend()

        plt.savefig('Results/0_Data/Train_Test_Split.png')
        plt.close()

        # Plot CV
        fig, axs = plt.subplots(self.cv.get_n_splits(), figsize=(15,6), sharex=True)

        fig.suptitle('CV Folds')

        for i, (train_index, test_index) in enumerate(self.cv.split(self.X_train)):
            axs[i].plot(self.X_train['date'], self.X_train['close'], color = 'k')
            axs[i].plot(self.X_train['date'][train_index], self.X_train['close'][train_index], label = 'Train')
            axs[i].plot(self.X_train['date'][test_index], self.X_train['close'][test_index], label = 'Test')
            axs[i].yaxis.set_visible(False)

        plt.savefig('Results/0_Data/CV_Folds.png')
        plt.close()

    def SelectModel(self, names, models, models_grid):

        print('=============================================================================')
        print('                          SELECTING BEST MODEL')
        print('=============================================================================')

        # Deleting date column
        self.df.drop(['date'], axis=1, inplace=True)
        self.X_train.drop(['date'], axis=1, inplace=True)
        self.X_train_val.drop(['date'], axis=1, inplace=True)
        self.X_val.drop(['date'], axis=1, inplace=True)
        self.X_test.drop(['date'], axis=1, inplace=True)

        Best_model_Tstart = time.time()

        # Results
        model_selection_results_path = 'Results/1_Model_Selection/Results'
        os.makedirs(model_selection_results_path, exist_ok=True)
        # Models
        model_selection_models_path = 'Results/1_Model_Selection/Models'
        os.makedirs(model_selection_models_path, exist_ok=True)
            
        # Searching best model
        for name, model, grid in zip(names, models, models_grid):
            print('Seeking best {} model...'.format(name))
            RS = RandomizedSearchCV(estimator=model,
                                    param_distributions=grid,
                                    cv=self.cv,
                                    n_iter=10,
                                    scoring=self.metric,
                                    random_state=self.seed,
                                    n_jobs=-1,
                                    verbose=3)
            
            RS.fit(self.X_train, self.y_train)

            best_params = RS.best_params_
            best_score = RS.best_score_
            best_model = RS.best_estimator_

            # Making validation tests
            best_model.fit(self.X_train_val, self.y_train_val)

            y_pred = best_model.predict(self.X_val)

            # Confusion Matrix
            conf_matrix = confusion_matrix(self.y_val, y_pred)

            # Classification Report
            class_rep = classification_report(self.y_val, y_pred, zero_division=0)

            print('Best {} model FOUND!!!'.format(name))
            
            # Printing Results step by step
            print('MODEL: {}'.format(name))
            print('Params: {}'.format(best_params))
            print('   SCORE: {}'.format(best_score))
            print('   CONFUSSION MATRIX: \n{}'.format(conf_matrix))
            print('   CLASSIFICATION REPORT: \n{}'.format(class_rep))

            # Dumping Results in file and saving models
            results_file = 'Results/1_Model_Selection/Results/Results_{}.txt'.format(name)
            with open(results_file, 'w') as f:
                f.write('MODEL: {}\n'.format(name))
                f.write('Params: {}\n'.format(best_params))
                f.write('   SCORE: {}\n'.format(best_score))
                f.write('   CONFUSSION MATRIX: \n{}\n'.format(conf_matrix))
                f.write('   CLASSIFICATION REPORT: \n{}\n'.format(class_rep))
                f.write('\n')
                f.close()

            model_file = 'Results/1_Model_Selection/Models/Best_{}.pkl'.format(name)
            with open(model_file, 'wb') as file:
                pickle.dump(best_model, file)
                file.close()

        Best_model_Tend = time.time()

        time_spent = Best_model_Tend - Best_model_Tstart

        print('Execution Time: {}'.format(time_spent))

    def FeatureSelection(self, model_name):

        print('=============================================================================')
        print('                          FEATURE SELECTION')
        print('=============================================================================')

        # Deleting date column
        self.df.drop(['date'], axis=1, inplace=True)
        self.X_train.drop(['date'], axis=1, inplace=True)
        self.X_train_val.drop(['date'], axis=1, inplace=True)
        self.X_val.drop(['date'], axis=1, inplace=True)
        self.X_test.drop(['date'], axis=1, inplace=True)

        Feature_selection_Tstart = time.time()

        # Loading model
        #Create Folders
        model_selection_results_path = 'Results/2_Feature_Selection'
        os.makedirs(model_selection_results_path, exist_ok=True)

        # Loading Model
        model_path = 'Results/1_Model_Selection/Models/Best_{}.pkl'.format(model_name)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            f.close()

        # Train model
        print('Training model....')
        model.fit(self.X_train, self.y_train)
        print('Model trained!')

        print('Selecting best features...')
        # Feature importance (Permutation Importance)
        importance = permutation_importance(estimator=model, X=self.X_train, y=self.y_train, scoring=self.metric, n_repeats=10,
                                            n_jobs=-1, random_state=self.seed)
        
        # Features importances
        ft_importances = pd.Series(data=importance.importances_mean, index=self.X_train.columns.to_numpy()).sort_values(ascending=False)

        # Plotting feature importances
        ft_importances.plot(kind='bar')
        plt.xlabel('Features')
        plt.ylabel('Importance')

        # Save plot
        file_path = 'Results/2_Feature_Selection/feature_importances.png'
        plt.savefig(file_path)
        plt.close()

        # Selecting only features with importances over mean

        mean = ft_importances.mean()

        self.best_feat = ft_importances.index[:75]
        best_num_feat = len(self.best_feat)

        # Keep selected features
        self.X_train = self.X_train[self.best_feat]
        self.X_train_val = self.X_train_val[self.best_feat]
        self.X_val = self.X_val[self.best_feat]
        self.X_test = self.X_test[self.best_feat]

        # Getting Results
        best_feat_score = cross_validate(estimator=model, X=self.X_train, y=self.y_train, scoring=self.metric,
                                         cv=self.cv, n_jobs=-1)['test_score'].mean()
        
        model.fit(self.X_train_val, self.y_train_val)

        y_pred = model.predict(self.X_val)

        # Confusion Matrix
        conf_matrix = confusion_matrix(self.y_val, y_pred)

        # Classification Report
        class_rep = classification_report(self.y_val, y_pred, zero_division=0)
        

        print('Best features selected!')
        Feature_selection_Tend = time.time()

        time_spent = Feature_selection_Tend - Feature_selection_Tstart
        print('Execution time: {}'.format(time_spent))

        # Saving Results

        feature_selection_results_path = 'Results/2_Feature_Selection/Results_75.txt'

        with open(feature_selection_results_path, 'w') as f:
            f.write('=============================================================================\n')
            f.write('                            FEATURE SELECTION\n')
            f.write('=============================================================================\n')
            f.write('Number of features selected: {}\n'.format(best_num_feat))
            f.write('Features selected: \n{}\n'.format(self.best_feat))
            f.write('Score: {}\n'.format(best_feat_score))
            f.write('Execution time: {}\n'.format(time_spent))
            f.write('Confussion Matrix:\n{}\n'.format(conf_matrix))
            f.write('Classification Report:\n{}\n'.format(class_rep))
            f.close()

    def HyperparametersTuning(self, model_name, features, grid):

        print('=============================================================================')
        print('                         HYPERPARAMETERS TUNING')
        print('=============================================================================')

        # Deleting date column
        self.df.drop(['date'], axis=1, inplace=True)
        self.X_train.drop(['date'], axis=1, inplace=True)
        self.X_train_val.drop(['date'], axis=1, inplace=True)
        self.X_val.drop(['date'], axis=1, inplace=True)
        self.X_test.drop(['date'], axis=1, inplace=True)

        Hyper_Tstart = time.time()
        
        # Define searching space
        self.hyper_space = grid

        # Keep features selected in feature selection
        self.best_feat = features

        self.X_train = self.X_train[self.best_feat]
        self.X_train_val = self.X_train_val[self.best_feat]
        self.X_val = self.X_val[self.best_feat]
        self.X_test = self.X_test[self.best_feat]

        # Define objective function
        def objective(params):
            if model_name == 'LR':
                model = LogisticRegression(**params)
            elif model_name == 'ET':
                model = ExtraTreesClassifier(**params)
            elif model_name == 'RF':
                model = RandomForestClassifier(**params)
            elif model_name == 'XGB':
                model = XGBClassifier(**params)

            score = cross_validate(model, self.X_train, self.y_train, cv=self.cv,
                                   scoring=self.metric, n_jobs=-1)['test_score'].mean()
            return -score

        # Search best hyperparameters
        trials = hyperopt.Trials()
        print('Seeking for best hyperparameters......')
        self.best_hyper = hyperopt.fmin(fn = objective, space = self.hyper_space, algo = hyperopt.tpe.suggest, max_evals = 1000, trials = trials,
                                        rstate=np.random.default_rng(self.seed))
        self.best_hyper_score = -min(trials.losses())
        print('Best hyperparameters found!')

        print('Best hyperparameters:\n{}\n'.format(self.best_hyper))
        print('Score: {}'.format(self.best_hyper_score))

        Hyper_Tend = time.time()
        time_spent = Hyper_Tend - Hyper_Tstart
        print('Execution time: {}'.format(time_spent))
        
        # Saving results
        Best_hyper_path = 'Results/3_Hyperparameters_Tunning'
        os.makedirs(Best_hyper_path, exist_ok=True)
        with open('Results/3_Hyperparameters_Tunning/Hyper_Tun_Results.txt', 'w') as f:
            f.write('=============================================================================\n')
            f.write('                          HYPERPARAMETERS TUNING\n')
            f.write('=============================================================================\n')
            f.write('Execution time: {}\n'.format(time_spent))
            f.write('Score: {}\n'.format(self.best_hyper_score))
            f.write('Best hyperparameters: \n{}\n'.format(self.best_hyper))
            f.close()

    def FinalModel(self, model_name, model, features):

        print('=============================================================================')
        print('                                  FINAL MODEL')
        print('=============================================================================')

        # Deleting date column
        self.df.drop(['date'], axis=1, inplace=True)
        self.X_train.drop(['date'], axis=1, inplace=True)
        self.X_train_val.drop(['date'], axis=1, inplace=True)
        self.X_val.drop(['date'], axis=1, inplace=True)
        self.X_test.drop(['date'], axis=1, inplace=True)

        Final_Tstart = time.time()
        
        # Keep features selected in feature selection
        self.best_feat = features

        self.X_train = self.X_train[self.best_feat]
        self.X_train_val = self.X_train_val[self.best_feat]
        self.X_val = self.X_val[self.best_feat]
        self.X_test = self.X_test[self.best_feat]
        
        print('Training final model......')

        # Train model
        model.fit(self.X_train, self.y_train)
        print('DONE!')

        Final_Tend = time.time()
        time_spent = Final_Tend - Final_Tstart
        print('Execution time: {}'.format(time_spent))

        # Obtain pred
        self.y_pred = model.predict(self.X_test)

        # RESULTS
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.class_rep = classification_report(self.y_test, self.y_pred, zero_division=0)

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.bal_accuracy = balanced_accuracy_score(self.y_test, self.y_pred)
        
        self.recall = recall_score(self.y_test, self.y_pred, average='macro')
        self.precision = precision_score(self.y_test, self.y_pred, average='macro')
        self.f1 = f1_score(self.y_test, self.y_pred, average='macro')

        print('Model: {}'.format(model_name))
        print('Confusion Matrix: \n{}'.format(self.confusion_matrix))
        print('Classification Report: \n{}'.format(self.class_rep))
        print('Accuracy: {}\n'.format(self.accuracy))
        print('Balanced accuracy: {}\n'.format(self.bal_accuracy))
        print('Recall: {}\n'.format(self.recall))
        print('Precision: {}\n'.format(self.precision))
        print('f1: {}\n'.format(self.f1))

        # Saving Results and Trained Model
        final_model_path = 'Results/4_Final_Model'
        os.makedirs(final_model_path, exist_ok=True)
        # Results
        with open('Results/4_Final_Model/Final_Results.txt', 'w') as f:
            f.write('=============================================================================\n')
            f.write('                          FINAL MODEL\n')
            f.write('=============================================================================\n')
            f.write('Execution time: {}\n'.format(time_spent))
            f.write('Model: {}\n'.format(model_name))
            f.write('Confusion Matrix: \n{}\n'.format(self.confusion_matrix))
            f.write('Classification Report: \n{}\n'.format(self.class_rep))
            f.write('Accuracy: {}\n'.format(self.accuracy))
            f.write('Balanced accuracy: {}\n'.format(self.bal_accuracy))
            f.write('Recall: {}\n'.format(self.recall))
            f.write('Precision: {}\n'.format(self.precision))
            f.write('F1: {}\n'.format(self.f1))
            f.close()
        #Trained model
        final_model_file = 'Results/4_Final_Model/Final_Model.pkl'
        with open(final_model_file, 'wb') as file:
            pickle.dump(model, file)
            file.close()