"""Processing of pickled objects from studying scripts
"""

import pickle
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


with open('params.pkl','rb') as f:
    data = pickle.load(f)

df = pd.DataFrame.from_dict(data).T

df.index.name = 'p_ratio'
df.columns = ['e_field_props','score']
df[['amplitude','freq','bias']] = df.e_field_props.tolist()
df[['score','baseline']] = df.score.tolist()

# print(df[df.score<1.1*df.baseline].head())
plt.subplot(1,2,1)
sns.scatterplot(data=df[df.score<df.baseline][['amplitude','bias']])
plt.subplot(1,2,2)
sns.scatterplot(data=df[df.score<df.baseline][['freq']])
plt.show()
