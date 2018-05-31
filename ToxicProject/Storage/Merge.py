import pandas as pd
import csv

#This code takes test.csv and test_label.csv
#and creates a new file called testMerged.csv
#It contains data from both tables formated correctly
#although code for removing the -1 data will be made soon
def main():
    temp_test = pd.read_csv("test.csv")
    test_label = pd.read_csv("test_labels.csv")

    temp_test["toxic"] = test_label["toxic"]
    temp_test["severe_toxic"] = test_label["severe_toxic"]
    temp_test["obscene"] = test_label["obscene"]
    temp_test["threat"] = test_label["threat"]
    temp_test["insult"]= test_label["insult"]
    temp_test["identity_hate"] = test_label["identity_hate"]
    temp_test.to_csv("testMerged.csv", encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)

        
if __name__ == "__main__":
    main()
