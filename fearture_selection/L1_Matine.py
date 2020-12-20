import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.manifold import LocallyLinearEmbedding


def mutual_mutual(data,label,k=290):
    model_mutual= SelectKBest(mutual_info_classif, k=k)
    new_data=model_mutual.fit_transform(data, label)
    return new_data
	
def elasticNet(data,label,alpha =np.array([0.1])): #np.array([0.01, 0.02, 0.03,0.04, 0.05, 0.06, 0.07, 0.08,0.09, 0.1]))
    enetCV = ElasticNetCV(alphas=alpha,l1_ratio=0.001,max_iter=2000).fit(data,label)
    enet=ElasticNet(alpha=enetCV.alpha_,l1_ratio=0.001,max_iter=2000)
    enet.fit(data,label)
    mask = enet.coef_ != 0
    new_data = data[:,mask]
    return new_data,mask
	
def selectFromExtraTrees(data,label):
    clf = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=None, 
                              class_weight=None)
    clf.fit(data,label)
    importance=clf.feature_importances_ 
    model=SelectFromModel(clf,prefit=True)
    new_data = model.transform(data)
    return new_data,importance
 
def lassodimension(data,label,alpha=np.array([0.00005])):#The alpha value range is 0.001, 0.002, 0.005, 0.01, 0.015, 0.02
    lassocv=LassoCV(cv=5, alphas=alpha).fit(data, label)
    x_lasso = lassocv.fit(data,label)
    mask = x_lasso.coef_ != 0 
    new_data = data[:,mask] 
    return new_data,mask 

def LLE(data,n_components=290):
    embedding = LocallyLinearEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed