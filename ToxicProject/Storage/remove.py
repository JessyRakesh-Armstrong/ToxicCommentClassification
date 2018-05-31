import pandas as pd
import csv
import numpy as np

original = pd.read_csv("test_preprocessed_full.csv")
x = original.values
new_list = list()
for line in range(len(x)):
    if x[line][2] != -1:
        new_list.append(x[line])
new = pd.DataFrame(data=new_list, columns=['id','comment_text','toxic',
                                           'severe_toxic','obscene','threat',
                                           'insult','identity_hate'])
new.to_csv("test_preprocessed_final.csv", encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
