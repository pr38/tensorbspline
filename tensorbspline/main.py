import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.exceptions import NotFittedError
import functools 

def knotbuilder(OneDdarray,knot1 = None,knot2 = None):

    if knot1 != None and knot2 != None:
        return np.piecewise(OneDdarray,np.logical_and([OneDdarray >= knot1],[OneDdarray < knot2]),[1, 0])
    
    if knot1 != None and knot2 == None:
        return np.piecewise(OneDdarray,[OneDdarray < knot1],[1, 0])
    
    if knot1 == None and knot2 != None:
        return np.piecewise(OneDdarray,[OneDdarray >= knot2],[1, 0])

def get_polynomial(OneDdarray, degree):

    output = np.polynomial.polynomial.polyvander(OneDdarray,degree)
    output = output.squeeze()[:,1:]
    return output

def get_RowWiseKroneckerProduct(array1,array2):

    if (type(array1) == np.ndarray) and (type(array1) == np.ndarray) == True:
        output = sparse.dok_matrix((array1.shape[0],(array1.shape[1]*array2.shape[1])))
        for i in range(0,array1.shape[0]):
            output[i] = np.kron(array1[i],array2[i])
        return output
    
    else:
        if (type(array1) == np.ndarray) == True:
            array1 = sparse.csr_matrix(array1)
        if (type(array2) == np.ndarray) == True:
            array2 = sparse.csr_matrix(array2)
        output = sparse.dok_matrix((array1.shape[0],(array1.shape[1]*array2.shape[1])))
        for i in range(0,array1.shape[0]):
            output[i] = sparse.kron(array1[i],array2[i])
        return output

class SplineBase( BaseEstimator, TransformerMixin):
    def __init__(self, 
            n_bin=10,
            polynomial_degrees = 3,
            sparse = False
             ):
        self.n_bin = n_bin
        self.polynomial_degrees = polynomial_degrees
        self.sparse = sparse
        
        self.knot_dict = {}
        self.is_fitted = False
        
    def get_knots(self,X):
        
        for col_pos in range(1,X.shape[1]):
            _ , bins_raw = np.histogram(X[:,col_pos], self.n_bin)
            
            bins_raw_unique = np.unique(bins_raw)
            
            bins = bins_raw_unique[1:-1]
            
            #saveing bins for transform stage
            self.knot_dict[col_pos] = bins        
    
    def check_is_fitted(self):
        if self.is_fitted != True:
            raise NotFittedError("""
            estimator is not fitted yet. Call 'fit' with appropriate arguments before using this estimator
            """)

    def fit(self,X, y= None):
        X = check_array(X)
        
        self.get_knots(X)

        self.is_fitted = True
    
        return self
    
    def get_splines(self,X):
        splines = []

        for col_pos in range(1,X.shape[1]):
            bins = self.knot_dict[col_pos]
            knots = []
            previous = None
            
            for bin_ in bins:
                if previous == None:
                    knot_ = knotbuilder(X[:,col_pos],knot1=bin_)
                
                else:
                    knot_ = knotbuilder(X[:,col_pos],knot1=previous,knot2 = bin_)

                previous = bin_
                knots.append(knot_.reshape(-1,1))


            knot_ = knotbuilder(X[:,col_pos],knot1=None,knot2 = previous)
            knots.append(knot_.reshape(-1,1))
            
            stepwisematrix = np.hstack(knots)
            
            polinomialmatrix = get_polynomial(X[:,col_pos],self.polynomial_degrees)
            
            bspline_matrix = np.hstack((stepwisematrix, polinomialmatrix))
            
                                    
            if self.sparse == True:
                bspline_matrix = sparse.dok_matrix(bspline_matrix)
                
            splines.append(bspline_matrix)
            
        return splines

class BSpline(SplineBase):
    def transform(self,X, y= None):
        self.check_is_fitted()
        X = check_array(X)
        
        splines = self.get_splines(X)


        if type(y) != type(None):
                           
            if self.sparse == False:
                return np.hstack(splines), y
            if self.sparse == True:
                return sparse.hstack(splines), y

        if type(y) == type(None):

            if self.sparse == False:
                return np.hstack(splines)
            
            if self.sparse == True:
                return sparse.hstack(splines)


class TensorBSplines(SplineBase):
    def transform(self,X, y= None):
        self.check_is_fitted()
        X = check_array(X)
        
        splines = self.get_splines(X) 

        if type(y) != type(None):
            return functools.reduce(get_RowWiseKroneckerProduct , splines), y

        if type(y) == type(None):
            return functools.reduce(get_RowWiseKroneckerProduct , splines)
                           
