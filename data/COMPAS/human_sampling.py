import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from datetime import datetime
import sys
np.set_printoptions(threshold = sys.maxsize)

inputfile = "labeled_data.csv"
outputfile = "sampled_humans.csv"


df = pd.read_csv(inputfile, index_col=0)

# 
df['hate_speech'] = df['hate_speech'] / df['count']
df['offensive_language'] = df['offensive_language'] / df['count']
df['neither'] = df['neither'] / df['count']

# print(df)

def sampler(hate_speech, offensive_language, neither) -> float :
    return np.random.choice([0,1,2], 1, p=list((hate_speech, offensive_language, neither)))[0]
nhumans = int(sys.argv[1])
for i in range(1, nhumans+1):
    df['h'+str(i)] = df.apply(lambda x : sampler(x['hate_speech'], x['offensive_language'], x['neither']), axis=1)


print(df)
df.to_csv(outputfile)


