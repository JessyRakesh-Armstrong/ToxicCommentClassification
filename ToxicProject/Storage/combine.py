import pandas as pd
import numpy as np
import csv

original = pd.read_csv("test_preprocessed.csv")
labels = pd.read_csv("test_labels.csv")
x = original.values
for line in range(len(x)):
    
