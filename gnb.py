# -*- coding: utf-8 -*-
"""
Spyder Editor

Gaussian Naive Bayes classifier
"""

import pandas as pd
import numpy as np

class gnb:
    
    '''
    Class for Gaussian Naive Bayes classifier
    '''
    
    def __init__(self):
        
        self.X = None
        self.y = None
        self.labels = None
        self.cat_probs = None
        self.means = None
        self.vars = None

    def fit(self, X, y):
    
        '''
        Fit Gaussian Naive Bayes classifier
        
        X, a pandas dataframe of data
        y, a 1D pandas dataframe of labels
    
        Returns nothing
        '''
    
        # Place data in object
        self.X = X
        self.y = y
        
        # Extract labels
        self.labels = sorted(pd.unique(y.values.flatten()))
        
        # Create arrays to hold parameters
        self.cat_probs = np.zeros([len(self.labels)])
        self.means = np.zeros([len(self.labels),X.shape[1]])
        self.vars = np.zeros([len(self.labels),X.shape[1]])
        
        # Work with numpy instead of pandas
        X_vals = X.values
        y_vals = y.values
        
        # Calculate marginal Gaussian distribution parameters
        for k in range(len(self.labels)):
            idx = (y_vals == self.labels[k]).flatten()
            self.cat_probs[k] = sum(idx) / len(y_vals)
            self.means[k] = np.mean(X_vals[idx,:],0)
            self.vars[k] = np.var(X_vals[idx,:],0)
            
    def predict(self,X):
        
        '''
        Predict using Naive Bayes classifier
        
        X, a pandas dataframe of data
        
        Returns predicted classes for each observation in X. "fit" method
        must be performed first
        '''

        X_vals = X.values
        prior_probs = np.zeros([X.shape[0],X.shape[1],len(self.labels)])
        
        # Compute prior probability elements
        for k in range(len(self.labels)):
            prior_probs[:,:,k] = 1 / (2 * np.pi * self.vars[k]) \
                           * np.exp(-np.square(X_vals - self.means[k]) \
                           / (2*self.vars[k]))
        
        # Compute posterior probabilities
        post_probs = self.cat_probs * np.product(prior_probs,1)
        
        # Apply MAP rule ot make classification as list
        map_result = np.argmax(post_probs,1)
        
        # Convert MAP result to class prediction
        prediction = list()
        for i in range(X_vals.shape[0]):
            prediction.append(self.labels[map_result[i]])
            
        # Convert prediction list to numpy
        prediction = np.reshape(prediction, (-1, 1))
        
        # Convert prediction numpy to as pandas and return
        return(pd.DataFrame(prediction, columns = ['diagnosis']))