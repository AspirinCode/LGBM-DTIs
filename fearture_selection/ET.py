import numpy as np
import pandas as pd
from sklearn.preprocessing import scale,StandardScaler
import utils.tools as utils
from L1_Matine import selectFromExtraTrees

data_train=pd.read_csv('GPCR.csv')
data_=np.array(data_train)
data=data_[:,1:]
label=data_[:,1]
shu=scale(data)	
data_2,importance=selectFromExtraTrees(shu,label)
shu=data_2
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('GPCR_ET_100.csv')