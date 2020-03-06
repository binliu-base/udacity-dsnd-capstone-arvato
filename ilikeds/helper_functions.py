import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 2020

def split_dataset(df, threshold=0.25):
    """ Splits data into two subsets based on the missing values per row.

    Args:
    - df (DataFrame): DataFrame to be split
    - threshold (float): threshold as decision criteria for splitting

    Returns:
    - DataFrame: DataFrame with a smaller percentage of missing values than the threshold 
    - DataFrame: DataFrame with a higher percentage of missing values than the threshold
    """
    nans_per_row = df.isnull().sum(axis=1)
    above_th = df[ nans_per_row / df.shape[1] <= threshold]
    below_th = df[ nans_per_row / df.shape[1] > threshold]

    return above_th, below_th


def split_date(p_data, p_column = 'EINGEFUEGT_AM', p_newColName = None):   
    """ Feature 'EINGEFUEGT_AM' contains timestamp-type data, We will split it into multiple time related new features.

    """

    v_new = p_column if p_newColName is None else p_newColName    

    p_data[f'{v_new}_Year']           = p_data[p_column].dt.year # The year of the datetime
    p_data[f'{v_new}_Month']          = p_data[p_column].dt.month # The month as January=1, December=12
    p_data[f'{v_new}_Day']            = p_data[p_column].dt.day # The day of the datetime
    p_data[f'{v_new}_Quarter']        = p_data[p_column].dt.quarter # The quarter of the date
    p_data[f'{v_new}_DaysInMonth']    = p_data[p_column].dt.days_in_month # The number of days in the month
    p_data[f'{v_new}_IsMonthStart']   = p_data[p_column].dt.is_month_start.apply(lambda x: 1 if x else 0)   # Flag for first day of month
    p_data[f'{v_new}_IsMonthEnd']     = p_data[p_column].dt.is_month_end.apply(lambda x: 1 if x else 0)     # Flag for last day of the month
    p_data[f'{v_new}_IsQuarterStart'] = p_data[p_column].dt.is_quarter_start.apply(lambda x: 1 if x else 0) # Flag for first day of a quarter
    p_data[f'{v_new}_IsQuarterEnd']   = p_data[p_column].dt.is_quarter_end.apply(lambda x: 1 if x else 0)   # Flag for last day of a quarter
    p_data[f'{v_new}_IsYearStart']    = p_data[p_column].dt.is_year_start.apply(lambda x: 1 if x else 0)    # Flag for first day of a year
    p_data[f'{v_new}_IsYearEnd']      = p_data[p_column].dt.is_year_end.apply(lambda x: 1 if x else 0)      # Flag for last day of a year    
    p_data[f'{v_new}_Season']         = p_data[f'{v_new}_Month'].apply(lambda x:      0 if x in [1, 2, 12] # 'winter'
                                                                                 else 1 if x in [3, 4, 5]  # 'spring'
                                                                                 else 2 if x in [6, 7, 8]  # 'summer'
                                                                                 else 3 )                  # 'fall'    

def check_features(eda, feat_type='categorical'):
    """  Access features of a specific type.
    Args:
    - feat_type (str): feature type (categorical, mixed, numeric, ordinal)

    Returns: None
    """
    
    for x in eda.feat_info.index:
        f, t = x, eda.feat_info.loc[x]['type']
        try:
            if(t == feat_type):
                print(f, eda.data.loc[: , f].unique())
        except KeyError:
            print(f + ' was eliminated already')


def plot_boxplot(data, feats, n_cols= 3, figsize= (25, 25)):
    """ Draw a box plot to show distributions with respect to features.

    Args:
    - data:  Dataset for plotting
    - feats: a list of features
    - n_cols: Number of features displayed per line
    - figsize: Size of the figure

    Returns: None
    """

    fig = plt.figure(figsize= figsize)        
    n_rows = int(len(feats)/n_cols) + 1

    for i in range(n_rows -1):
        for j in range(n_cols):
            fig.add_subplot(n_rows , n_cols, i*n_cols + j + 1)                     
            f = feats[i*n_cols + j]
            sns.boxplot(x = data[f] )    
            plt.title(f'Quartile distribution for feature: {f}')  


def do_pca(eda, n_components):
    ''' Transforms data using PCA to create n components

    Args: 
        eda - EDA object instance
        n_components - int - the number of principal components to create

    Returns:  None
    '''

    eda.pca = PCA(n_components, random_state=12)
    eda.X_pca = eda.pca.fit_transform(eda.data_scaled)


# plot_feature_comparison1
def plot_2feats_comparison(data, feats, figsize = (25, 25)):
    ''' Comparing two features in the same dataset

    Args: 
        data:  Dataset for plotting 
        feats: a list of features           
        figsize: Size of the figure

    Returns:  None
    '''    

    fig = plt.figure(figsize= figsize)    
    n_rows =  len(feats)
    count = (x +1 for x in range(2*n_rows))    
    for row in feats :
        f1, f2 = row
        fig.add_subplot(n_rows, 2, next(count))
        data[f1].dropna().value_counts().apply(lambda x: x/data[f1].dropna().shape[0]).plot(kind ='bar', label =f1, rot=0, fontsize=20 )
        plt.legend(fontsize=20 )
        
        fig.add_subplot(n_rows, 2, next(count))
        data[f2].dropna().value_counts().apply(lambda x: x/data[f2].dropna().shape[0]).plot(kind ='bar', label =f2, rot=90, fontsize=20 )
        plt.legend(fontsize=20)            

# plot_feature_comparison2
def plot_feats_comparison(data1, data2, feats, fig_height=4, fig_aspect=0.8):
    ''' Comparing features between two datasets

    Args: 
        data1:  Dataset 1 for plotting 
        data2:  Dataset 2 for plotting         
        feats: a list of features           
        fig_height: height of the figure
        fig_aspect: Aspect ratio of each facet, so that fig_aspect * fig_height gives the width of each facet in inches.

    Returns:  None
    '''    

    feats_data1 =pd.Series(feats, index = feats).apply(lambda x:  data1[x].value_counts()/data1[x].dropna().shape[0])    
    feats_data2 =pd.Series(feats, index = feats).apply(lambda x:  data2[x].value_counts()/data2[x].dropna().shape[0])        

    class_df = pd.DataFrame()
    for x in feats:  
        try:
            t1 =pd.DataFrame(feats_data1.stack()[x], columns=['Percent'])
            t1['DATASET'] = 'Azdias'
            t1['Value'] = t1.index.astype(int)
            t1['Feature'] = x                            
            t1.set_index(['Feature','DATASET','Value'], inplace =True)
            class_df = pd.concat([class_df, t1])

            t2 =pd.DataFrame(feats_data2.stack()[x], columns=['Percent'])
            t2['DATASET'] = 'Customers'
            t2['Value'] = t2.index.astype(int)
            t2['Feature'] = x                                
            t2.set_index(['Feature','DATASET','Value'], inplace =True)
            class_df = pd.concat([class_df, t2])
        except TypeError as e:
            print(x)

    n_rows = int(len(feats)/5) + 1
                
    for i in range(n_rows):
        if i < n_rows -1:
            d = class_df.reset_index().set_index('Feature').loc[feats[5*i: 5*i+5]].reset_index()
        else:
            d = class_df.reset_index().set_index('Feature').loc[feats[5*i: ]].reset_index()        

        try:
            sns.catplot(
                    x = 'Value', y ='Percent' , 
                    data= d,
                    col="Feature",
                    hue = 'DATASET',
                    sharey=False,                     
                    sharex=False,    
                    kind="bar",
                    height=fig_height,
                    aspect=fig_aspect,
                    palette = sns.color_palette("muted")
                    )        
        except ValueError as e:
            pass

# def scree_plot(pca, label):
def scree_plot(eda):
    ''' Creates a scree plot associated with the principal components 
    
    Args: 
        eda - EDA object instance        
            
    Returns: None
    '''
    num_components=len(eda.pca.explained_variance_ratio_)    
    ind = np.arange(num_components)
    vals = eda.pca.explained_variance_ratio_
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Accumulated Variance Explained")

    # 100 components
    plt.hlines(y=cumvals[99], xmin=0, xmax=99, color='red', linestyles='-',zorder=3)
    plt.vlines(x=99, ymin=0, ymax=cumvals[99], color='red', linestyles='-',zorder=4)

    # 200
    plt.hlines(y=cumvals[num_components-1], xmin=0, xmax=num_components-1, color='red', linestyles='-',zorder=5)
    plt.vlines(x=num_components-1, ymin=0, ymax=cumvals[num_components-1], color='red', linestyles='-',zorder=6)    
    
    plt.title(f'PCA Analysis ({eda}))')   

def get_kmeans_score(data, center):
    ''' Returns the kmeans score regarding SSE for points to centers
    Args:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    Returns:
        score - the SSE score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)

    # Then fit the model to your data using the fit method
    model = kmeans.fit(data)

    # Obtain a score related to the model fit
    score = np.abs(model.score(data))
    
    # if p_plot:
    #     labels = model.predict(data)
    #     plot_data(data, labels)

    return model, round(score, 3)

def plot_cluster_comparison(preds_c, preds_a):
    """ Comparing components between two clusters 
    Args: 
        preds_c - Value of predicted principal components of dataset Customers
        preds_a - Value of predicted principal components of dataset Azdias
            
    Returns: None

    """
    
    def prepare_data(preds, label):
        counts_df = pd.DataFrame(
            {
            'cluster' :  np.unique(preds),
            'count':  np.bincount(preds), 
            'percent': np.bincount(preds)/preds.shape[0] * 100,
            },
        )
        counts_df['type'] = label
        return counts_df    

    counts_c = prepare_data(preds_c, 'Customers')
    print()    
    counts_a = prepare_data(preds_a, 'Azdias')

    frame = [counts_c, counts_a]
    result_df = pd.concat(frame, keys = ['cluster', 'count', 'percent'], axis = 0)
    
    ax = sns.catplot( x = 'cluster',y = 'percent', data= result_df,
                    hue = 'type',
                    kind="bar",
                    sharey=True,
                    height=5,
                    aspect=1.618,
                    palette = sns.color_palette("muted")
                    )

    ax.set_xticklabels(result_df.cluster.unique(), rotation=0)   
    ax.set(title=f'Global Cluster VS Customer Cluster') 
    
    return counts_c, counts_a

def list_component(eda, top_n_comps):
    """ Listing the top N and the bottom N features of a given component
    Args: 
        eda - EDA object instance        
        top_n_comps            
    Returns: None

    """    
    listing = pd.DataFrame({
                        'Features':list(eda.data.columns),
                        'Weights':eda.pca.components_[top_n_comps]}).sort_values('Weights', axis=0, ascending=False).values.tolist()

    return listing[:5]+['^^^HEAD','TAILvvv']+listing[-5:]  


def build_roc_auc(model_dict, param_grid, X_train, y_train):
    '''
    Function for calculating auc and roc
    Args: 
        model_dict - dict of model map
        param_grid - dict or list of dictionaries      
        X_train - the training data
        y_train - the training response values (must be 0 or 1)
    Returns: None
    prints the roc auc score
    '''
    
    for k, clf in model_dict.items():
            grid = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc', cv=10)
            grid.fit(X_train, y_train)
            print(f'Model: {k},  Best ROC AUC score:  {grid.best_score_}')           


def save_runs(run_name, model, model_type, space, trials, best, n_iter, preds_test):
    run_dic = {
        'run_name': run_name,
        'model' : model,
        'model_type': model_type,
        'space': space,
        'trails': trials,
        'best':  best,
        'n_iters':  n_iter,
        'preds_test': preds_test
    }
    
    file = open( run_name+'.pkl',  'wb')
    pickle.dump(run_dic, file)