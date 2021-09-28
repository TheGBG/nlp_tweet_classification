"""
Quick look at the data and how clean/dirty it is
"""
import pandas as pd

datapath = "data/raw/train.csv"
raw_data = pd.read_csv(datapath)

raw_data.shape
raw_data.columns

raw_data.head()

raw_data.isna().sum(axis=0) / raw_data.shape[0]

# We will be using text and target. It makes no sense to use id, location
# has plenty of NA, and keyword... well. Lets keep it for now

columns_to_drop = ['id', 'location']
raw_data.drop(columns=columns_to_drop, axis=0, inplace=True)

# Now if we're planning to use keyword we will need to drop NA's as it
# don't make sense imputing in this kind of problem. Lets see what we 
# acctually have in that column