import sklearn
from sklearn.datasets import make_circles
import pandas as pd

n_samples = 1000

X,y = make_circles(n_samples,noise=0.03,random_state=10)

circles = pd.DataFrame({"X1":X[:,0] , "X2":X[:,1], "label":y})
 

