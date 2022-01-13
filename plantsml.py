import mlxtend as mx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
import pandas as pd
import os

# Creating a list of lists
def read_lists(name):
    with open(name, "r") as f:
        lines = f.readlines()
    states = []
    for line in lines:
        states.append(line.strip().split(',')[1:])
    return states

# Opening the database using the read_lists function
plantas = read_lists("C:/Users/jl_sa/Desktop/plants.data")
# Applying the TransactionEncoder to get a boolean array
te = TransactionEncoder()
te_plants = te.fit(plantas).transform(plantas)
# Transforming the boolean array to a pandas data frame
df = pd.DataFrame(te_plants.astype('int'), columns=te.columns_)
# Applying the apriori method (mlxtend library)
fp = apriori(df, min_support=0.1, use_colnames=True)
# Applying the association_rules method (mlxtend library)
rules = association_rules(fp, metric="confidence", min_threshold=0.5)
# Creating a boolean Series to filter data by confidence >= 0.5
confidence = rules['confidence'] >= 0.5
# Getting the filtered and sorted by confidence data frame
confidence_high = rules[confidence].sort_values(by='confidence')
print(confidence_high)
# Saving to a .txt file
s_confidence = confidence_high.to_string()
path = "C:/Users/jl_sa/Desktop/Facul"
if os.path.exists(path+"s_confidence.txt"):
    pass
else:
    path2= os.path.join(path, "s_confidence.txt")
    arquivo = open(path2,"a")
    arquivo.write(s_confidence)
    arquivo.close()