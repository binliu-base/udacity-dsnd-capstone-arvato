import numpy as np
import pandas as pd
import ast

from sklearn.preprocessing import Imputer, StandardScaler

import helper_functions as h

class EDA(object):

    def __init__(self, data, feat_info, label = 'Azdias'):

        self.data = data
        self.feat_info = None
        self.pca = None
        self.X_pca = None
        self.label = label
        self.feat_info = self.build_feat_info(feat_info)

    def __repr__(self):        
        return self.label
    
    def _preprocess(self, feat_info):
        self.build_feat_info(feat_info)
        

    def build_feat_info(self, feat_info):

        if 'is_drop' not in feat_info.columns:            
            feat_info['is_drop'] = 0

        if 'action' not in feat_info.columns:            
            feat_info['action'] = 0

        feats = feat_info.index.tolist()
        stats_df = self.data[feats].describe()
        
        return pd.DataFrame({
            'type': feat_info.type,
            'unknow': feat_info.unknow,    
            'is_drop': feat_info.is_drop,    
            'action': feat_info.action,                
            'n_nans' : self.data[feats].isnull().sum(),
            'percent_of_nans' :  round(self.data[feats].isnull().sum()/self.data[feats].shape[0], 3),
            'value_distinct': pd.Series([self.data[c].unique().shape[0] for c in feats], index = feats),
            'value_count': stats_df.loc['count'],
            'value_mean': stats_df.loc['mean'],
            'value_std': stats_df.loc['std'],
            'value_min': stats_df.loc['min'],
            'value_max': stats_df.loc['max'],
            'value_Q1': stats_df.loc['25%'],
            'value_Q3': stats_df.loc['75%'],                        
            'value_IRQ': stats_df.loc['75%'] - stats_df.loc['25%'],                                    
        }, index = feat_info.index)

    def missing2nan(self):  
        ''' Converts all the unknows values in the dataset to NaN.
        Args: None
        Returns: None
        '''
        n_nans_bef= self.data.isnull().sum().sum()

        # for col in self.data.columns:
        # for col in self.feat_info['unknow'].dropna().index:
        for col in self.data.columns:
            if col in self.feat_info['unknow'].dropna().index:
                unknows = ast.literal_eval(self.feat_info.loc[col]['unknow'])
                self.data[col] = self.data[col].mask(self.data[col].isin(unknows), other=np.nan)

        n_nans_aft = self.data.isnull().sum().sum()    
        change = (n_nans_aft - n_nans_bef)/n_nans_bef *100            

        print(f'Number of missing values in {self.label}:')
        print(f'Before converstion is {n_nans_bef}')
        print(f'Ater converstion IS {n_nans_aft}')
        print('Increase in missing values: {0:.2f} % '.format(change))


    def re_encoding(self, p_feat_name):
        ''' Re-encoding a feature
        Args: 
            p_feat_name:  the feature name of re-encoding
        Returns: None

        '''        

        def re_encoding_OST_WEST_KZ():
            self.data['OST_WEST_KZ'] = self.data['OST_WEST_KZ'].map({'W': 1, 'O': 0})
            # self.feat_info.loc['OST_WEST_KZ','action'] = h.action_dic[3]              
            
        def re_encoding_CAMEO_DEUG_2015():
            self.data['CAMEO_DEUG_2015'].replace({'X': np.nan}, inplace= True)
            self.data['CAMEO_DEUG_2015'] = self.data['CAMEO_DEUG_2015'].astype('float64')
            # self.feat_info.loc['CAMEO_DEUG_2015','action'] = h.action_dic[3]                          

        def re_encoding_CAMEO_INTL_2015():
            self.data['CAMEO_INTL_2015'].replace({'XX': np.nan}, inplace= True)          
            self.data['CAMEO_INTL_2015'] = self.data['CAMEO_INTL_2015'].astype('float64')   
            # self.feat_info.loc[CAMEO_INTL_2015,'action'] = h.action_dic[3]                          

        def re_encoding_CAMEO_DEU_2015():
            self.data.CAMEO_DEU_2015.replace({'XX': -1}, inplace =True)
            codes, uniques = pd.factorize(self.data.CAMEO_DEU_2015.value_counts().index.tolist())
            mapping_CAMEO_DEU_2015 = dict(zip(uniques,codes))
            self.data.CAMEO_DEU_2015 = self.data.CAMEO_DEU_2015.map(mapping_CAMEO_DEU_2015).astype('float64')    
            # self.feat_info.loc['CAMEO_DEUG_2015','action'] = h.action_dic[3]                          

        func_dict= {
            'OST_WEST_KZ': re_encoding_OST_WEST_KZ,
            'CAMEO_DEUG_2015': re_encoding_CAMEO_DEUG_2015,
            'CAMEO_INTL_2015': re_encoding_CAMEO_INTL_2015,   
            # 'CAMEO_DEU_2015': re_encoding_CAMEO_DEU_2015,                               
            # 'LP_LEBENSPHASE_GROB': re_encoding_LP_LEBENSPHASE_GROB,                                                   

        } 

        func_dict[p_feat_name]()
        self.feat_info.loc[p_feat_name,'action'] = h.action_dic[3]  

    def split_PRAEGENDE_JUGENDJAHRE(self):

        def convert_pj_to_move(val):
            """
            Converts value of feature PRAEGENDE_JUGENDJAHRE to a MOVEMENT value.

            INPUT:
            - val (int): original value

            OUTPUT:
            - int: converted value (0: Mainstream, 1: Avantgarde)
            """
            
            result = val
            if (val in [1,3,5,8,10,12,14]):
                result = 0 # Mainstream
            elif (val in [2,4,6,7,9,11,13,15]):
                result = 1 # Avantgarde
            return result

        map_DECADE = {
            1   : 0,  2   : 0,  #40s            
            3   : 1,  4   : 1,  #50s            
            5   : 2,  6   : 2,  7  : 2,  #60s            
            8   : 3,  9   : 3,  #70s            
            10   : 4, 11  : 4,  12   : 4,  13   : 4,  #80s            
            14   : 5, 15  : 5,  #90s
        }

        # Converts value of feature PRAEGENDE_JUGENDJAHRE to a DECADE value. 
        # converted value (0: 40s, 1: 50s, 2: 60s, 3: 70s, 4: 80s, 5: 90s)           
        self.data['PRAEGENDE_JUGENDJAHRE_SPLIT_DECADE'] = self.data['PRAEGENDE_JUGENDJAHRE'].map(map_DECADE).astype(float)
        self.data['PRAEGENDE_JUGENDJAHRE_SPLIT_MOVEMENT'] = self.data['PRAEGENDE_JUGENDJAHRE'].apply(lambda x: convert_pj_to_move(x))

    def split_CAMEO_INTL_2015(self):

        self.data['CAMEO_INTL_2015_SPLIT_WEALTH'] = self.data['CAMEO_INTL_2015'].apply(lambda x: np.floor(pd.to_numeric(x)/10))        
        self.data['CAMEO_INTL_2015_SPLIT_LIFE_STAGE'] = self.data['CAMEO_INTL_2015'].apply(lambda x: pd.to_numeric(x)%10)   

    def split_LP_LEBENSPHASE_GROB(self):

        map_FAMILY = {
            1   : 0,  2   : 0,  3: 0, #SINGLE            
            4   : 1,  5   : 1,  #SINGLE-COUPLES 
            6   : 2,  #SINGLE-PARENTS
            7   : 3,  #SINGLE-FAMILY 
            8   : 4,  #FAMILY
            9   : 5, 10: 5, 11: 5, 12: 5,  #MULITPERSON-HOUSEHOLDS
        }

        map_AGE = {
            1 :   0, 9 : 0 , 11: 0, #YOUNGER
            2 :   1, 4:  1, 10: 1, 12: 1, #HIGHER
            3 :   np.NaN, 5: np.NaN, 6: np.NaN, 7: np.NaN,  8: np.NaN,   

        }

        map_INCOME = {
            1 :   0, 2 : 0 , 4: 0, 7: 0, 10: 0, 9: 0 ,#LOW-AND-AVERAGE
            3 :  1, 5 : 1, 8: 1, 11: 1, 12: 1, #HIGH
            6 :   np.NaN,

        }

        self.data['LP_LEBENSPHASE_GROB_SPLIT_FAMILY'] = self.data['LP_LEBENSPHASE_GROB'].map(map_FAMILY).astype(float)
        self.data['LP_LEBENSPHASE_GROB_SPLIT_AGE'] = self.data['LP_LEBENSPHASE_GROB'].map(map_AGE).astype(float)        
        self.data['LP_LEBENSPHASE_GROB_SPLIT_INCOME'] = self.data['LP_LEBENSPHASE_GROB'].map(map_INCOME).astype(float)  

    def split_EINGEFUEGT_AM(self):
        self.data['EINGEFUEGT_AM'] = pd.to_datetime(self.data['EINGEFUEGT_AM'])
        h.split_date(self.data, 'EINGEFUEGT_AM')    

    def split_mixed_feat(self, p_feat_name):
        ''' Handling mixed type features
        Args: 
            p_feat_name:  the mixed feature name
        Returns: None

        '''           

        func_dict= {
            'PRAEGENDE_JUGENDJAHRE': self.split_PRAEGENDE_JUGENDJAHRE,
            'CAMEO_INTL_2015': self.split_CAMEO_INTL_2015,
            'LP_LEBENSPHASE_GROB': self.split_LP_LEBENSPHASE_GROB,
            'EINGEFUEGT_AM': self.split_EINGEFUEGT_AM,                        
            # 'CAMEO_DEU_2015': self.split_CAMEO_DEU_2015,
        }     

        func_dict[p_feat_name]() 
        self.feat_info.loc[p_feat_name ,['is_drop', 'action']] = 1, h.action_dic[5]   
        self.data.drop(p_feat_name, axis=1, inplace=True)                        
   

    def update_stats(self):
        ''' Collecting statistical information of the dataset and update the feat_info table
        Args: 
            args:  list of statistical metrics
        Returns: None
        '''           
        feats = self.feat_info.loc[self.feat_info.is_drop == 0].index

        stats_df = self.data[feats].describe()

        feat_info = pd.DataFrame({
            'type': self.feat_info.loc[feats].type,
            'unknow': self.feat_info.loc[feats].unknow,    
            'is_drop': self.feat_info.loc[feats].is_drop.astype(int),    
            'action': self.feat_info.action,                 
            'n_nans' : self.data[feats].isnull().sum().astype(int),
            'percent_of_nans' : round(self.data[feats].isnull().sum()/self.data[feats].shape[0], 3).astype(float),
            'value_distinct': pd.Series([self.data[c].unique().shape[0] for c in feats], index = feats).astype(int),
            'value_count': stats_df.loc['count'].astype(int),
            'value_mean': stats_df.loc['mean'],
            'value_std': stats_df.loc['std'],
            'value_min': stats_df.loc['min'],
            'value_max': stats_df.loc['max'],
            'value_Q1': stats_df.loc['25%'],
            'value_Q3': stats_df.loc['75%'],                        
            'value_IRQ': stats_df.loc['75%'] - stats_df.loc['25%'],                                    
                                   
        }, index = feats)  

        self.feat_info.loc[feats] = feat_info

    def clean_outlier(self, feats):
        ''' Createing  outliers of the dataset
        Args: 
            feats:  list of feature names
        Returns: None
        '''                   

        def make_outlier_map():  

            # numeric features
            MIN_GEBAEUDEJAHR_MIN = self.feat_info.loc['MIN_GEBAEUDEJAHR'].value_Q1 - 5
            MIN_GEBAEUDEJAHR_MAX = self.feat_info.loc['MIN_GEBAEUDEJAHR'].value_Q3 + 5

            EINGEZOGENAM_HH_JAHR_MIN = self.feat_info.loc['EINGEZOGENAM_HH_JAHR'].value_Q1 - self.feat_info.loc['EINGEZOGENAM_HH_JAHR'].value_IRQ
            EINGEZOGENAM_HH_JAHR_MAX = self.feat_info.loc['EINGEZOGENAM_HH_JAHR'].value_Q3 + self.feat_info.loc['EINGEZOGENAM_HH_JAHR'].value_IRQ

            ANZ_HAUSHALTE_AKTIV_MIN = self.feat_info.loc['ANZ_HAUSHALTE_AKTIV'].value_Q1 - self.feat_info.loc['ANZ_HAUSHALTE_AKTIV'].value_IRQ
            ANZ_HAUSHALTE_AKTIV_MAX = self.feat_info.loc['ANZ_HAUSHALTE_AKTIV'].value_Q3 + self.feat_info.loc['ANZ_HAUSHALTE_AKTIV'].value_IRQ    

            ANZ_PERSONEN_MIN = self.feat_info.loc['ANZ_PERSONEN'].value_Q1 - self.feat_info.loc['ANZ_PERSONEN'].value_IRQ
            ANZ_PERSONEN_MAX = self.feat_info.loc['ANZ_PERSONEN'].value_Q3 + self.feat_info.loc['ANZ_PERSONEN'].value_IRQ     

            KBA13_ANZAHL_PKW_MIN = self.feat_info.loc['KBA13_ANZAHL_PKW'].value_Q1 - self.feat_info.loc['KBA13_ANZAHL_PKW'].value_IRQ
            KBA13_ANZAHL_PKW_MAX = self.feat_info.loc['KBA13_ANZAHL_PKW'].value_Q3 + self.feat_info.loc['KBA13_ANZAHL_PKW'].value_IRQ                           

            # categorical features
            # CAMEO_DEU_2015_MIN = self.feat_info.loc['CAMEO_DEU_2015'].Q1 - self.feat_info.loc['CAMEO_DEU_2015'].IQR
            # CAMEO_DEU_2015_MAX = self.feat_info.loc['CAMEO_DEU_2015'].Q3 + self.feat_info.loc['CAMEO_DEU_2015'].IQR            

            D19_KONSUMTYP_MIN =  self.feat_info.loc['D19_KONSUMTYP'].value_Q1 - self.feat_info.loc['D19_KONSUMTYP'].value_IRQ
            D19_KONSUMTYP_MAX =  self.feat_info.loc['D19_KONSUMTYP'].value_Q3 + self.feat_info.loc['D19_KONSUMTYP'].value_IRQ

            outlier_map = { 
                    'MIN_GEBAEUDEJAHR':  {'MIN': MIN_GEBAEUDEJAHR_MIN ,  'MAX': MIN_GEBAEUDEJAHR_MAX},
                    'EINGEZOGENAM_HH_JAHR':    {'MIN': EINGEZOGENAM_HH_JAHR_MIN, 'MAX': EINGEZOGENAM_HH_JAHR_MAX},
                    'ANZ_HAUSHALTE_AKTIV':    {'MIN': ANZ_HAUSHALTE_AKTIV_MIN, 'MAX': ANZ_HAUSHALTE_AKTIV_MAX},                    
                    'ANZ_PERSONEN':    {'MIN': ANZ_PERSONEN_MIN, 'MAX': ANZ_PERSONEN_MAX},                                        
                    'KBA13_ANZAHL_PKW':    {'MIN': KBA13_ANZAHL_PKW_MIN, 'MAX': KBA13_ANZAHL_PKW_MAX},                                                            
                    # 'CAMEO_DEU_2015':  {'MIN': CAMEO_DEU_2015_MIN,  'MAX': CAMEO_DEU_2015_MAX},
                    'D19_KONSUMTYP':    {'MIN': D19_KONSUMTYP_MIN, 'MAX': D19_KONSUMTYP_MAX},
                    }

            return outlier_map

        def remove_outliers(x):
            lim=outlier_map[x]
            if 'MIN' in lim.keys():
                self.data[x] = self.data[x].fillna( self.data[x].mean()).apply(lambda k: lim['MIN'] if k < lim['MIN'] else k)

            if 'MAX' in lim.keys():
                self.data[x] = self.data[x].fillna( self.data[x].mean()).apply(lambda k: lim['MAX'] if k > lim['MAX'] else k) 

        outlier_map = make_outlier_map()                

        for x in feats:
            print(f'Cleaning outliers for {x}  ...')            
            remove_outliers(x)

        self.feat_info.loc[x ,'action'] = h.action_dic[7]   

    def data_pipeline(self, thr_row_missing = 0.25, clean_rows =True ): 
        ''' All in one, cleaning data
        Args: 
            thr_row_missing:  Threshold of remove rows with more NaN values
            clean_rows: Decide whether to delete rows
        Returns: None
        '''            

        # Delete undefined, multiple missing values and duplicate features
        print(f'Step 1: Delete undefined, multiple missing values and duplicate features ...')
        print(f'Before cleaning, Number of columns is {self.data.shape[1]} in {self.label} ')
        feats_todrop = pd.read_csv('feats_dropped.csv', header=None)        
        feats_todrop = pd.DataFrame(feats_todrop)[0].values.tolist()
        self.data.drop(columns= feats_todrop, axis = 1,inplace =True)
        print(f'After cleaning, Number of columns is {self.data.shape[1]} in {self.label} ')        

        #  Convert missing and unknow values
        print(f'Step 2: Convert missing and unknow values ...')
        self.missing2nan()  

        # Delete the rows with more NaN values
        if clean_rows:
            print(f'Step 3: Delete the rows with more NaN values ...')        
            _, rows_droped = h.split_dataset(self.data, threshold=0.25)
            n_droped_rows = rows_droped.shape[0]

            print(f'Before cleaning, Number of rows is {self.data.shape[1]} in {self.label} ')            
            self.data.drop(index = rows_droped.index, inplace =True)
            print(f'After cleaning, Number of columns is {self.data.shape[1]} in {self.label} ')        
            print(f'  {n_droped_rows} lines deleted!')            


        print(f'Step 4: Re-encoding features...')        
        feats_encoding = ['OST_WEST_KZ',
                          'CAMEO_DEUG_2015',
                          'CAMEO_INTL_2015',
                          'EINGEFUEGT_AM']

        for x in feats_encoding:
            print(f'   Re-encoding: {x} ...')                           
            self.re_encoding(x)

        # Split mixed features
        print(f'Step 5: Split mixed features ...')
        mixed_feats = ['CAMEO_INTL_2015', 'LP_LEBENSPHASE_GROB', 'PRAEGENDE_JUGENDJAHRE']
        for x in mixed_feats:
            print(f'   Spliting: {x} ...')                           
            self.split_mixed_feat(x)

        feats_splited = ['CAMEO_INTL_2015_SPLIT_WEALTH', 
                         'CAMEO_INTL_2015_SPLIT_LIFE_STAGE',
                         'LP_LEBENSPHASE_GROB_SPLIT_FAMILY',
                         'LP_LEBENSPHASE_GROB_SPLIT_AGE',
                         'LP_LEBENSPHASE_GROB_SPLIT_INCOME',
                         'PRAEGENDE_JUGENDJAHRE_SPLIT_DECADE',
                         'PRAEGENDE_JUGENDJAHRE_SPLIT_MOVEMENT']

        feat_info_split = self.build_split_feat_info(feats_splited)   
        self.feat_info= pd.concat([self.feat_info, feat_info_split], sort = False)

        self.update_stats()

        print(f'Step 6: Handling outliers ...')  
        outlier_feats = ['MIN_GEBAEUDEJAHR', 'EINGEZOGENAM_HH_JAHR']      
        self.clean_outlier(outlier_feats)

        # Estimating NaN values with median
        print(f'Step 7: Estimating NaN values with median ...')                
        imputer = Imputer(strategy='median')
        self.data_imputed = pd.DataFrame(imputer.fit_transform(self.data),  columns=self.data.columns)

        print(f'Step 8: Feature scaling ...')                        
        self.data_scaled = pd.DataFrame(StandardScaler().fit_transform(self.data_imputed), columns=self.data.columns)

        print(f'Data Cleaning done !')                        