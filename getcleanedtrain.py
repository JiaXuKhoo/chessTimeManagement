import numpy as np
import pandas as pd

data = pd.read_csv("classifier_train_data.csv")
data = data[data["status"] == "ok"].drop(columns=["status", "error_msg", "ref_nodes"])
data.to_csv("cleaned_train_data.csv", index=False)
fens = list(data["fen"].values)

with open("ok_30k_fens.txt", "w") as output:
    for fen in fens:
        output.write(fen + "\n")