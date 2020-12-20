##LGBM-DTIs

LGBM-DTIs：Predicting drug-target interactions using LightGBM with protein multi-information and molecular structure.


###LGBM-DTIs uses the following dependencies:
* MATLAB2014a
* python 3.6 
* numpy
* pandas
* scikit-learn
* imblearn 


###Guiding principles:

**The dataset file contains the gold standard dataset, Kuang dataset and network dataset.

**feature extraction:
1) Evolutionary-based features: PsePSSM.m is the implementation of PsePSSM. 
2) Sequence-based features: PAAC.py is the implementation of PseAAC.
3) Structural-based features: Structure.py is the implementation of structural information based on SPIDER3.
   
** feature selection:
   Lasso.py represents the Lasso.
   Elastic_net.py represents the elastic net.
   ET.py represents the extra trees.
   IG.py represents the information gain.
   LLE.py represents the locally linear embedding.
   MI.py represents the mutual information.
   
** data preprocessing:
   SMOTE.R is the implementation of SMOTE. 
   AllKNN.py is the implementation of AllKNN. 
   ENN.py is the implementation of edited nearest neighbours. 
   OSS.py is the implementation of one-sided selection. 
   RUS.py is the implementation of random undersampling.

** Classifier:
   LightGBM.py is the implementation of LightGBM.
   AdaBoost.py is the implementation of AdaBoost.
   DT.py is the implementation of decision tree. 
   GBM.py is the implementation of gradient boosting machine.
   KNN.py is the implementation of K-nearest neighbor. 
   LR.py is the implementation of logistic regression. 
   NB.py is the implementation of NB.
   RF.py is the implementation of random forest. 
   SVM.py is the implementation of support vector machine.

