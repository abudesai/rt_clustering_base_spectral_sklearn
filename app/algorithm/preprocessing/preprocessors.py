
import numpy as np, pandas as pd
import sys 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


class TypeCaster(BaseEstimator, TransformerMixin):  
    def __init__(self, vars, cast_type):
        super().__init__()
        self.vars = vars
        self.cast_type = cast_type
        
    def fit(self, X, y=None): return self
        

    def transform(self, data):  
        data = data.copy()
        applied_cols = [col for col in self.vars if col in data.columns] 
        for var in applied_cols: 
            data[var] = data[var].apply(self.cast_type)
        return data



class FloatTypeCaster(TypeCaster):  
    ''' Casts float features as object type if they are not already so.
    This is needed when some categorical features have values that can inferred as numerical.
    This causes an error when doing categorical feature engineering. 
    '''
    def __init__(self, num_vars):
        super(FloatTypeCaster, self).__init__(num_vars, float)



class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns, selector_type='keep'):
        self.columns = columns
        self.selector_type = selector_type
        
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):   
        
        if self.selector_type == 'keep':
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == 'drop':
            dropped_cols = [col for col in X.columns if col in self.columns]  
            X = X.drop(dropped_cols, axis=1)      
        else: 
            raise Exception(f'''
                Error: Invalid selector_type. 
                Allowed values ['keep', 'drop']
                Given type = {self.selector_type} ''')   
        return X


class CustomSVD(BaseEstimator, TransformerMixin):    
    def __init__(self, id_col):
        self.id_col = id_col
        self.svd = None
        self.fitted_cols = None
        self.reduced_cols = None
        self.min_cols = 5
    
    def fit(self, X, y=None):
        self.fitted_cols = [c for c in X.columns if c != self.id_col]
        num_cols = len(self.fitted_cols)
        n_components = num_cols if num_cols < self.min_cols else self.min_cols + int(np.sqrt(num_cols - self.min_cols))
        if n_components < num_cols: 
            self.reduced_cols = [f'svd_{i}' for i in range(n_components)]
            self.svd = TruncatedSVD(n_components=n_components)
            self.svd.fit(X[self.fitted_cols])
        else: 
            pass
        return self
    
    def transform(self, data): 
        if not self.svd is None: 
            transformed = self.svd.transform(data[self.fitted_cols])
            transformed = pd.DataFrame(transformed, columns=self.reduced_cols)
            transformed.insert(0, self.id_col, data[self.id_col])
        else: 
            transformed = data
        return transformed
    

class XSplitter(BaseEstimator, TransformerMixin): 
    def __init__(self, id_col):
        self.id_col = id_col
        self.ids = None
        self.ids2idx = {}
        self.idx_col = "__idx__"
    
    def fit(self, data): 
        self.ids = data[[self.id_col]].drop_duplicates()   
        self.ids[self.idx_col] = np.arange(self.ids.shape[0])
        return self
    
    def transform(self, data):        
        X_cols = [ col for col in data.columns if col != self.id_col ]  
        data2 = data.merge(self.ids, on=[self.id_col])  
        return { 'X': data2[X_cols], 'ids': data2[self.id_col]   , "idxs": data2[self.idx_col] }
