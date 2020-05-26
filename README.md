# Proyecto_Estadistica
Para ejecutar el aplicativo de aprendizaje supervisado se debe garantizar tener instaladas las siguientes librerias:

El formato de carga de información es excel, y se debe realizar con la sentencia pd.read_excel, siendo pd el alias de la libreria de pandas.

Librerias a instalar

import statsmodels.api as sm
import seaborn as sns
import pylab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, classification_report,f1_score,recall_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from scipy import stats


