import numpy as np
import pandas as pd
from sklearn.preprocessing import scale,StandardScaler
from L1_Matine import LLE
import utils.tools as utils

data_train=pd.read_csv(r'GPCR.csv')
data_=np.array(data_train)
data=data_[:,1:]
label=data_[:,0]
shu=scale(data)
new_X=LLE(shu,n_components=462)
shu=new_X
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('GPCR_LLE_290.csv')
