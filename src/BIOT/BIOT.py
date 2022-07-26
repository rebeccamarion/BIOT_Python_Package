#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions and classes for the BIOT module

@author: Rebecca Marion
"""

import pandas as pd 
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import math

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

###############################################################################

# Functions and classes

def Get_W_Lasso (X, Y, lam, fit_intercept = False):
    """
    Estimates multiple Lasso models (one for each column of Y).

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of d features (columns) used to explain the embedding
        
    Y : numpy.ndarray
        Embedding matrix containing m dimensions to be explained
        
    lam : float
        Sparsity hyperparameter
        
    fit_intercept : boolean
        If True, an intercept is estimated 
        (Default value = False)

    Returns
    -------
    numpy.ndarray
        Matrix of model weights (d features x m embedding dimensions)
    numpy.ndarray
        Vector of m model intercepts 
    """
    
    k = Y.shape[1]
    d = X.shape[1]
    W = np.zeros((d, k))
    w0 = np.zeros(k)
    # Fit Lasso for each dimension of Y
    for dim_index in range(k):
        model = Lasso(alpha = lam, fit_intercept = fit_intercept, max_iter = 5000)
        model.fit(X = X, y = Y[:, dim_index])
        W[:, dim_index] = model.coef_
        if fit_intercept:
            w0[dim_index] = model.intercept_
        
    return W, w0

def Global_L1_Norm (W):
    """
    Calculates the sum of L1 norms for each column of W.

    Parameters
    ----------
    W : numpy.ndarray
        Matrix of model weights
        
    Returns
    -------
    float
        Sum of L1 norms for each column of W
    """
    
    k = W.shape[1]
    norm_val = 0
    for dim_index in range(k):
        norm_val += np.linalg.norm(W[:, dim_index], ord = 1)
        
    return norm_val

def BIOT_Crit (X, Y, R, W, w0, lam):
    """
    Calculates the criterion value for the BIOT objective function.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of d features (columns) used to explain the embedding
        
    Y : numpy.ndarray
        Embedding matrix containing m dimensions to be explained
         
    R : numpy.ndarray
        Orthogonal transformation matrix (m x m)
        
    W : numpy.ndarray
        Matrix of model weights  (d features x m embedding dimensions)
        
    w0 : numpy.ndarray
        Vector of m model intercepts
        
    lam : float
        Sparsity hyperparameter
        
    Returns
    -------
    float
        Criterion value for the BIOT objective function
    """
    
    n = X.shape[0]
    diffs = (Y @ R) - (np.tile(w0, (n, 1)) + (X @ W))
    LS = np.linalg.norm(diffs)**2
    L1 = Global_L1_Norm(W)
    
    crit = ((1/(2*n)) * LS) + (lam * L1)
    
    return crit

def BIOT (X, Y, lam, max_iter = 500, eps = 1e-6, rotation = False, R = None, fit_intercept = False):
    """
    Fits a BIOT model.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of d features (columns) used to explain the embedding
        
    Y : numpy.ndarray
        Embedding matrix containing m dimensions to be explained
        
    lam : float
        Sparsity hyperparameter for BIOT
        
    max_iter : int
        Maximum number of iterations to run
        (Default value = 500)
    eps : float
        Convergence threshold
        (Default value = 1e-6)
    rotation : boolean
        If true, the transformation matrix is constrained to be a rotation matrix 
        (Default value = False)
    R : numpy.ndarray
        Optional orthogonal transformation matrix (if provided, R will not be optimized)
        (Default value = None)
    fit_intercept : boolean
        If True, an intercept is estimated 
        (Default value = False)
         
    Returns
    -------
    numpy.ndarray
        Orthogonal transformation matrix (m x m) 
    numpy.ndarray
        Matrix of model weights (d features x m embedding dimensions) 
    numpy.ndarray
        Vector of m model intercepts 
    """
    
    d = X.shape[1]
    n = X.shape[0]
    lam_norm = lam/np.sqrt(d)
    
    # If R is provided, get Lasso solution only
    if R is not None:
        YR = Y @ R
        W, w0 = Get_W_Lasso(X = X, Y = YR, lam = lam_norm, fit_intercept=fit_intercept)
    # Otherwise, run BIOT iterations
    else:
        # Init W
        W, w0 = Get_W_Lasso(X = X, Y = Y, lam = lam_norm, fit_intercept=fit_intercept)
        
        diff = math.inf
        iter_index = 0
        crit_list = [math.inf]
        
        while iter_index < max_iter and diff > eps:
            
            # UPDATE R
            u, s, v = np.linalg.svd((1/(2*n)) * Y.T @ (np.tile(w0, (n, 1)) + (X @ W)))
        
            # rotation matrix desired (counterclockwise)
            if rotation:
                sv = np.ones(len(s))
                which_smallest_s = np.argmin(s)
                sv[which_smallest_s] = np.sign(np.linalg.det(u @ v))
                R = u @ np.diag(sv) @ v
            # orthogonal transformation matrix desired
            else:
                R = u @ v
                
            # UPDATE W
            YR = Y @ R
            W, w0 = Get_W_Lasso(X = X, Y = YR, lam = lam_norm, fit_intercept = fit_intercept)
            
            # CHECK CONVERGENCE
            crit_list.append(BIOT_Crit(X = X, Y = Y, R = R, W = W, w0 = w0, lam = lam_norm))
            diff = np.absolute(crit_list[iter_index] - crit_list[iter_index + 1])
    
            iter_index += 1
            
       
    return R, W, w0

class BIOTRegressor(BaseEstimator, RegressorMixin):
    """ 
    BIOT class, inherits methods from BaseEstimator and RegressorMixin classes
    from sklearn.base
    """
    def __init__(self, lam = 1, R = None, rotation = False, fit_intercept = False, feature_names = None):
        """
        Parameters
        ----------
        lam : float
            Sparsity hyperparameter
            (Default value = 1)
        R : numpy.ndarray
            Optional orthogonal transformation matrix (if provided, R will not be optimized)
            (Default value = None)
        rotation : boolean
            If true, the transformation matrix is constrained to be a rotation matrix 
            (Default value = False)
        fit_intercept : boolean
            If True, an intercept is estimated 
            (Default value = False)
        feature_names : pandas.core.indexes.base.Index or numpy.ndarray
            Names of the features potentially used to explain the embedding 
            dimensions
            (Default value = None)
        """
        
        self.lam = lam
        self.R = R
        self.rotation = rotation
        self.fit_intercept = fit_intercept
        self.feature_names = feature_names
        
    def fit(self, X, Y):
        """
        Fit method for BIOTRegressor class.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame or numpy.ndarray
            Matrix of d features (columns) used to explain the embedding (training set)
            
        Y : pandas.core.frame.DataFrame or numpy.ndarray
            Embedding matrix containing m dimensions to be explained (training set)
            
        Returns
        -------
        object
            Fitted estimator
        """
       
        X = check_array(X)
        Y = check_array(Y)
        
        R, W, w0 = BIOT(X = X, Y = Y, lam = self.lam, rotation = self.rotation, R = self.R, fit_intercept = self.fit_intercept)
        self.R_ = R
        self.W_ = pd.DataFrame(W, index = self.feature_names)
        self.w0_ = w0
        
        return self
    
    def predict(self, X):
        """
        Predict method for BIOTRegressor class.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame or numpy.ndarray
            Matrix of d features (columns) used to explain the embedding (prediction set)
            
        Returns
        -------
        numpy.ndarray
            Predicted values for Y (Y = (w0 1^t + X W) R^t)
        """
        check_is_fitted(self)
        X = check_array(X)
        n = X.shape[0]
        W = check_array(self.W_)
        R = check_array(self.R_)
        intercept = np.tile(self.w0_, (n, 1))
        
        return (intercept + (X @ W)) @ R.T
    
class myPipe(Pipeline):
    """ 
    Class that inherits from the Pipeline class, making the BIOT solution R, W 
    and w0 accessible in the .best_estimator_ attribute of the GridSearchCV 
    class.
    """

    def fit(self, X, Y):
        """Calls last elements .R_, .W_ and .w0_ method.
        Based on the sourcecode for decision_function(X).
        Link: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/pipeline.py
        ----------

        Parameters
        ----------
        X : pandas.core.frame.DataFrame or numpy.ndarray
            Matrix of d features (columns) used to explain the embedding (training set)
            
        Y : pandas.core.frame.DataFrame or numpy.ndarray
            Embedding matrix containing m dimensions to be explained (training set)
            
        Returns
        -------
        object
            Fitted estimator
        """

        super(myPipe, self).fit(X, Y)

        self.R_ = self.steps[-1][-1].R_
        self.W_ = self.steps[-1][-1].W_
        self.w0_ = self.steps[-1][-1].w0_
        
        
        return

def CV_BIOT (X_train, X_test, Y_train, lam_list, fit_intercept = False, num_folds=10, random_state = 1, R = None, rotation = False, scoring = 'neg_mean_squared_error'):
    """
    Cross-validated BIOT.
    
    Performs K-fold cross-validation to select the best lambda value for BIOT, 
    and estimates a final model using all data in X_train and Y_train and the 
    selected lambda value. Predictions are then calculated for X_test.

    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame or numpy.ndarray
        Matrix of d features (columns) used to explain the embedding (training set)
        
    X_test : pandas.core.frame.DataFrame or numpy.ndarray
        Matrix of d features (columns) used to explain the embedding (test set)
        
    Y_train : pandas.core.frame.DataFrame or numpy.ndarray
        Embedding matrix containing m dimensions to be explained (training set)
        
    lam_list : list
        List of lambda values to test during cross-validation 
        
    fit_intercept : boolean
        If True, an intercept is estimated 
        (Default value = False)
    num_folds : int
        Number of folds for K-fold cross-validation
        (Default value = 10)
    random_state : int
        Seed for reproducible results
        (Default value = 1)
    R : numpy.ndarray
        Optional orthogonal transformation matrix (if provided, R will not be optimized)
        (Default value = None)
    rotation : boolean
        If true, the transformation matrix is constrained to be a rotation matrix 
        (Default value = False)
    scoring : str
        Scoring function to use for the selection of the hyperparameter lambda.
        For all options, see the list of regression scoring functions at
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter .
        (Default value = 'neg_mean_squared_error')

    Returns
    -------
    numpy.ndarray
        Matrix of predicted values for Y, calculated based on the final model 
        (W, w0 and R) and X_test: (w0 1^t + X_test W) R^t
    pandas.core.frame.DataFrame
        Matrix of model weights (d features x m embedding dimensions) for the
        final model, feature names contained in the index (if available)
    numpy.ndarray
        Vector of m model intercepts for the final model
    numpy.ndarray
        Orthogonal transformation matrix (m x m) for the final model
    """
   
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns
        
    else:
        feature_names = np.array(range(X_train.shape[0]))
        
     
    # define the model pipeline
    pipe = myPipe([
         ('sc', StandardScaler()),
         ('BIOT', BIOTRegressor(R = R, rotation = rotation, fit_intercept = fit_intercept, feature_names = feature_names))
     ])        
    
    space = dict()
    space['regressor__BIOT__lam'] = lam_list
    
    # configure the cross-validation procedure
    cv = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    # define search
    estimator = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler(with_std = False))
    search = GridSearchCV(estimator = estimator, param_grid = space, scoring=scoring, cv=cv, refit=True)
    # execute search
    search.fit(X_train, Y_train)
    # get the best performing model fit on the whole training set
    best_model = search.best_estimator_
    # evaluate model on the hold out dataset
    Yhat = best_model.predict(X_test)
    
    
    W = best_model.regressor_.W_
    R = best_model.regressor_.R_
    w0 = best_model.regressor_.w0_
    
    return Yhat, W, w0, R


def calc_max_lam (X, Y):
    """
    Calculate the smallest lambda value resulting in an empty Lasso model.
    
    Before calculating this value, X is centered and scaled, and Y is centered.

    Parameters
    ----------
    X : pandas.core.frame.DataFrame or numpy.ndarray
        Matrix of d features (columns) used to explain the embedding
        
    Y : pandas.core.frame.DataFrame or numpy.ndarray
        Embedding matrix containing m dimensions to be explained
        
    Returns
    -------
    float
        Smallest lambda value resulting in an empty Lasso model
    """
    n = X.shape[0]
    sc = StandardScaler()
    X_norm = sc.fit_transform(X)
    sc = StandardScaler(with_std=False)
    Y_norm = sc.fit_transform(Y)
    max_lam = np.max(np.absolute(X_norm.T @ Y_norm))/n
    
    return max_lam