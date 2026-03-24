import pandas as pd
import sklearn
import sklearn.metrics as sklearn_metric
import warnings
from scipy.stats import kendalltau
import scipy
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)
import krippendorff
from irrCAC.raw import CAC

# 1] QWK
def qwk_score(y_true, y_pred, min, max):
  if np.all(np.array(y_true) == np.array(y_pred)):
     return 1.0
  return sklearn_metric.cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=list(range(min, max+1)))

# 2] GWET AC2 with two weights: quadratic and linear
def gwet_ac2_quadratic_score(y_true, y_pred, min, max):
  data = pd.DataFrame({'Rater1': y_pred, 'Rater2': y_true})
  raters2 = CAC(data, weights='quadratic', categories=list(range(min, max+1)))
  return raters2.gwet()['est']['coefficient_value']

def gwet_ac2_linear_score(y_true, y_pred, min, max):
  data = pd.DataFrame({'Rater1': y_pred, 'Rater2': y_true})
  raters2 = CAC(data, weights='linear', categories=list(range(min, max+1)))
  return raters2.gwet()['est']['coefficient_value']

# 3] RMSE from sklearn directly
def rmse_score(y_true, y_pred, min, max):
   return np.sqrt(sklearn_metric.mean_squared_error(y_true, y_pred))

# 4] ACC from sklearn directly
def acc_score(y_true, y_pred, min, max):
   return sklearn_metric.accuracy_score(y_true, y_pred)

# 5] Krippendorff alpha with 'interval'
def krippendorff_alpha_score(y_true, y_pred, min, max, level_of_measurement='interval'):
    """
    Computes Krippendorff's alpha between two raters: y_true and y_pred.

    Parameters:
        y_true (list or array): Ratings from Rater 1
        y_pred (list or array): Ratings from Rater 2
        level_of_measurement (str): 'nominal', 'ordinal', or 'interval'

    Returns:
        float: Krippendorff's alpha
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    # This is special case where we might have small number of essays and all have same value
    if np.all(np.array(y_true) == np.array(y_pred)):
     return 1.0

    # Construct a 2 x N array where each column is one item, each row is a rater
    data = np.array([y_true, y_pred])

    # Replace missing values with np.nan if needed — Krippendorff handles that
    return krippendorff.alpha(reliability_data=data, level_of_measurement=level_of_measurement)

# Pearson's r
def pearson_score(y_true, y_pred, min, max):
   return scipy.stats.pearsonr(y_true, y_pred)[0]

# ?] Cohen's d
def cohen_d_score(x, y, min, max):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)