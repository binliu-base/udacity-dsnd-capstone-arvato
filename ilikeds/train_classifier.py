import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedKFold, cross_val_score, learning_curve    
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import lightgbm as lgb
from skopt import BayesSearchCV
from hyperopt import hp, tpe,anneal, fmin, Trials

import eda 
import helper_functions as h



def search_hyperparameter(def_params, n_iter, cv, train_data,  train_targets):
    """ Returns the best hyper parameters, logging data and function to optimize
    Args:
        def_params: Dict of search space of hyper parameters 
        n_iter:  Maximum number of iterations
        cv:  Number of k-fold CV
        train_data:  The data to fit 
        train_targets: The target variable to try to predict

    Returns:
        best: the best hyper parameters
        trials: logging information of a test
        objective: function to optimize
    """    
    def objective(params, def_params=def_params, X=train_data, y=train_targets):

            # the function gets a set of variable parameters in "param"
            params = {x[0] : params[x[0]] if x[1]['dtype'] == 'float' else int(params[x[0]])  for x  in def_params.items()}   

            # we use this params to create a new LGBM Regressor
            clf = lgb.LGBMClassifier(
                    random_state=h.RANDOM_STATE, 
                    application='binary', 
                    class_weight='balanced',                    
                    cv =cv,
                    **params)            

            # and then conduct the cross validation with the same folds as before
            score =-cross_val_score(clf, X, y, scoring="roc_auc", n_jobs=-1).mean()

            return score

    # trials will contain logging information
    trials = Trials()

    space = {
        k: v['hpf'] for k, v in zip(def_params.keys(), def_params.values())
    }
    
    best = fmin(fn=objective, # function to optimize
            space=space, 
            algo=anneal.suggest, # optimization algorithm, hyperotp will select its parameters automatically
            max_evals=n_iter, # maximum number of iterations
            trials=trials, # logging
            rstate=np.random.RandomState(h.RANDOM_STATE), # fixing random state for the reproducibility
            )

    return best, trials, objective

def build_model(best, def_params):
    """    
    Args:
        best: the best hyper parameters    
        def_params: Dict of search space of hyper parameters 

    Returns:
       model (LightGBM LGBMClassifier): LGBMClassifier model object                
    """

    params = {x[0] : best[x[0]]   if x[1]['dtype'] == 'float' else int(best[x[0]])  for x  in def_params.items()}
        
    return lgb.LGBMClassifier(
                            random_state=h.RANDOM_STATE, 
                            application='binary',  
                            class_weight='balanced', 
                            **params,) 



def evaluate_model(model, objective, best, X_test, Y_test):
    """Prints multi-output classification results
    Args:
        model (pandas dataframe): the scikit-learn fitted model
        objective: function to optimize
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test classifications

    Returns: None
    """
    sa_test_score= - roc_auc_score(Y_test, model.predict(X_test))
    print("Best ROC {:.3f} params {}".format(objective(best), best))


def plot_result(trials):    
    def get_results(x):
        score = np.abs(x['result']['loss'])
        params = np.array([p for p in x['misc']['vals'].values()], dtype='float32')
        sa_results = np.append(arr=score , values=params)
        return sa_results        

    sa_results=np.array([    
        get_results(x)
    for x in trials] )
    
    sa_columns = ['score']
    sa_columns.extend(trials[0]['misc']['vals'].keys())    

    sa_results_df = pd.DataFrame(sa_results, columns=sa_columns)
    sa_results_df.plot(subplots=True,figsize=(10, 10))
    return sa_results_df

def save_model(model, model_filepath):
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to

    Returns: None
    """    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))



def main():

        eda_mailout_train = pd.read_pickle ('../ArvatoPrj_200222.liso/eda_mailout_train.full.pkl')        
        response = pd.read_pickle ('../ArvatoPrj_200222.liso/response.pkl')        

        random_state = 2020
        train_data, test_data, train_targets, test_targets = train_test_split(
            eda_mailout_train.data_scaled, 
            response, 
            test_size=0.20, 
            shuffle=True,
            random_state=random_state)

        train_data, test_data, train_targets, test_targets = pd.DataFrame(train_data), pd.DataFrame(test_data), pd.Series(train_targets), pd.Series(test_targets) 
        
        print('Building model...')
        model = build_model()

        
        # print('Training model...')
        
        # f = open(model_filepath, 'rb')
        # model = pickle.load(f)
        
        # print('Evaluating model...')
        # evaluate_model(model, test_data, test_targets)

        # print('Saving model...\n    MODEL: {}'.format(model_filepath))
        # save_model(model, model_filepath)

        # print('Trained model saved!')


if __name__ == '__main__':
    main()