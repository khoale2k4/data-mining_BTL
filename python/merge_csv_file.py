import pandas as pd

train = pd.read_csv("../data/train.csv")
features = pd.read_csv("../data/features.csv")
stores = pd.read_csv("../data/stores.csv")

df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left") \
          .merge(stores, on="Store", how="left")

df.to_csv('../data/walmart_recruiting_dataset.csv', index=False)