import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from L1_Matine import lassodimension

data_=pd.read_csv(r'GPCR_zongyangben.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((1476,1))#Value can be changed
label2=np.zeros((41364,1))
label=np.append(label1,label2)
shu=scale(data)

data_2,mask=lassodimension(shu,label)
shu=data_2

data_csv = pd.DataFrame(data=shu)
data_csv.to_csv(r'GPCR_Lasso.csv')
