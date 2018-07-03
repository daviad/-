#!/usr/bin python
#-*- coding:utf-8 -*-
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
 
np.random.seed(10)
n_estimator = 10
 
X, y = make_classification(n_samples=80000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#To avoid overfitting
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
 
def RandomForestLR():
	rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
	rf_enc = OneHotEncoder()
	rf_lr = LogisticRegression()
	rf.fit(X_train, y_train)
	rf_enc.fit(rf.apply(X_train))
	rf_lr.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
	y_pred_rf_lr = rf_lr.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
	fpr_rf_lr, tpr_rf_lr, _ = roc_curve(y_test, y_pred_rf_lr)
	auc = roc_auc_score(y_test, y_pred_rf_lr)
	print("RF+LR:", auc)
	return fpr_rf_lr, tpr_rf_lr
 
def GdbtLR():
	grd = GradientBoostingClassifier(n_estimators=n_estimator)
	grd_enc = OneHotEncoder()
	grd_lr = LogisticRegression()
	grd.fit(X_train, y_train)
	grd_enc.fit(grd.apply(X_train)[:, :, 0])
	grd_lr.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
	y_pred_grd_lr = grd_lr.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
	fpr_grd_lr, tpr_grd_lr, _ = roc_curve(y_test, y_pred_grd_lr)
	auc = roc_auc_score(y_test, y_pred_grd_lr) 
	print("GDBT+LR:", auc)
	return fpr_grd_lr, tpr_grd_lr
 
def Xgboost():
	xgboost = xgb.XGBClassifier(nthread=4, learning_rate=0.08,\
			n_estimators=50, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
	xgboost.fit(X_train, y_train)
	y_xgboost_test = xgboost.predict_proba(X_test)[:, 1]
	fpr_xgboost, tpr_xgboost, _ = roc_curve(y_test, y_xgboost_test)
	auc = roc_auc_score(y_test, y_xgboost_test)
	print("Xgboost:", auc)
	return fpr_xgboost, tpr_xgboost
 
def Lr():
	lm = LogisticRegression(n_jobs=4, C=0.1, penalty='l1')
	lm.fit(X_train, y_train)
	y_lr_test = lm.predict_proba(X_test)[:, 1]
	fpr_lr, tpr_lr, _ = roc_curve(y_test, y_lr_test)
	auc = roc_auc_score(y_test, y_lr_test)
	print("LR:", auc)
	return fpr_lr, tpr_lr
 
def XgboostLr():
	xgboost = xgb.XGBClassifier(nthread=4, learning_rate=0.08,\
	                            n_estimators=50, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
	xgb_enc = OneHotEncoder()
	xgb_lr = LogisticRegression(n_jobs=4, C=0.1, penalty='l1')
	xgboost.fit(X_train, y_train)
 
	xgb_enc.fit(xgboost.apply(X_train)[:, :])
	xgb_lr.fit(xgb_enc.transform(xgboost.apply(X_train_lr)[:, :]), y_train_lr)
	y_xgb_lr_test = xgb_lr.predict_proba(xgb_enc.transform(xgboost.apply(X_test)[:,:]))[:, 1]
	fpr_xgb_lr, tpr_xgb_lr, _ = roc_curve(y_test, y_xgb_lr_test)
	auc = roc_auc_score(y_test, y_xgb_lr_test)
	print("Xgboost + LR:", auc)
	return fpr_xgb_lr, tpr_xgb_lr
 
if __name__ == '__main__':
	fpr_rf_lr, tpr_rf_lr = RandomForestLR()
	fpr_grd_lr, tpr_grd_lr = GdbtLR()
	fpr_xgboost, tpr_xgboost = Xgboost()
	fpr_lr, tpr_lr = Lr()
	fpr_xgb_lr, tpr_xgb_lr = XgboostLr()
 
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
	plt.plot(fpr_grd_lr, tpr_grd_lr, label='GBT + LR')
	plt.plot(fpr_xgboost, tpr_xgboost, label='XGB')
	plt.plot(fpr_lr, tpr_lr, label='LR')
	plt.plot(fpr_xgb_lr, tpr_xgb_lr, label='XGB + LR')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()
 
	plt.figure(2)
	plt.xlim(0, 0.2)
	plt.ylim(0.8, 1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
	plt.plot(fpr_grd_lr, tpr_grd_lr, label='GBT + LR')
	plt.plot(fpr_xgboost, tpr_xgboost, label='XGB')
	plt.plot(fpr_lr, tpr_lr, label='LR')
	plt.plot(fpr_xgb_lr, tpr_xgb_lr, label='XGB + LR')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve (zoomed in at top left)')
	plt.legend(loc='best')
