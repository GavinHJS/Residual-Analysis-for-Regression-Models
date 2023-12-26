# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:45:56 2023

@author: Gavin
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.formula.api import ols
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
    
class ResidualAnalysis:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.residuals = y - model.predict(X)

    def summary_statistics(self):

        return {
            'mean': np.mean(self.residuals),
            'median': np.median(self.residuals),
            'std_dev': np.std(self.residuals)
        }

    def plot_residuals(self):

        plt.figure(figsize=(10, 6))
        plt.scatter(self.y, self.residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Observed Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Observed Values')
        plt.show()

    def check_normality(self):

        return stats.shapiro(self.residuals)

    def plot_histogram(self):

        sns.histplot(self.residuals, kde=True)
        plt.title('Histogram of Residuals')
        plt.show()

    def autocorrelation_check(self):

        plot_acf(self.residuals)
        plt.title('Autocorrelation plot')
        plt.show()

    def check_homoscedasticity(self):

        plt.scatter(self.model.predict(self.X), self.residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.show()


    def identify_outliers(self):

        z_scores = np.abs(stats.zscore(self.residuals))
        return np.where(z_scores > 3)[0]

    def check_linearity(self):

        for i, col in enumerate(self.X.columns):
            plt.figure(figsize=(10, 6))
            plt.scatter(self.X[col], self.residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel(col)
            plt.ylabel('Residuals')
            plt.title(f'Residuals vs {col}')
            plt.show()

    def multicollinearity_check(self):

        vif_data = pd.DataFrame()
        vif_data["feature"] = self.X.columns
        vif_data["VIF"] = [variance_inflation_factor(self.X.values, i) for i in range(len(self.X.columns))]
        return vif_data

    def durbin_watson_test(self):

        return durbin_watson(self.residuals)
    
    def leverage_analysis(self):

        influence = OLSInfluence(self.model)
        leverage = influence.hat_matrix_diag
        plt.figure(figsize=(10, 6))
        plt.scatter(leverage, self.residuals)
        plt.xlabel('Leverage')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Leverage')
        plt.show()

    def identify_influential_observations(self):

        influence = OLSInfluence(self.model)
        cooks_d = influence.cooks_distance[0]
        return np.where(cooks_d > 4 / len(self.X))[0]
    
    def breusch_pagan_test(self):

        _, pval, _, _ = het_breuschpagan(self.residuals, self.X)
        return pval
    
    def detailed_residual_distribution(self):

        plt.figure(figsize=(12, 6))
        sns.distplot(self.residuals, fit=stats.norm)
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Residual Distribution with Normal Fit')
        plt.show()
    
        stats.probplot(self.residuals, plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.show()
    
    def quantile_residual_plot(self):

        quantiles = np.percentile(self.residuals, range(0, 101, 5))
        sns.boxplot(data=self.residuals, orient='h', whis=1.5)
        plt.yticks(range(0, 101, 5), quantiles)
        plt.title('Boxplot of Residuals by Quantiles')
        plt.show()

    
    def plot_predicted_vs_actual(self):

        plt.figure(figsize=(10, 6))
        plt.scatter(self.model.predict(self.X), self.y, alpha=0.5)
        plt.plot(self.y, self.y, color='red')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Predicted vs Actual Values')
        plt.show()
        
        
if __name__ == "__main__":

    X, y = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)
    X = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    y = pd.Series(y, name='Target')

    model = LinearRegression().fit(X, y)


    X_with_constant = add_constant(X)

    ols_model = OLS(y, X_with_constant).fit()
    

    residual_analysis = ResidualAnalysis(ols_model, X_with_constant, y)
    print(ols_model.summary())
    
    print("Summary Statistics:", residual_analysis.summary_statistics())
    residual_analysis.plot_residuals()
    print("Normality Check:", residual_analysis.check_normality())
    residual_analysis.plot_histogram()
    residual_analysis.autocorrelation_check()
    residual_analysis.check_homoscedasticity()
    print("Outliers:", residual_analysis.identify_outliers())
    residual_analysis.check_linearity()
    print("Multicollinearity Check:", residual_analysis.multicollinearity_check())
    print("Durbin Watson Test:", residual_analysis.durbin_watson_test())
    residual_analysis.leverage_analysis()
    print("Influential Observations:", residual_analysis.identify_influential_observations())
    print("Breusch Pagan Test:", residual_analysis.breusch_pagan_test())
    residual_analysis.detailed_residual_distribution()
    residual_analysis.quantile_residual_plot()
    residual_analysis.plot_predicted_vs_actual()