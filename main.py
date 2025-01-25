from src.pricemodel.manager import *


df = pd.read_csv('data/sales_2021_on_geo.csv').sample(n=500, random_state = 92)
manager, df_with_predictions = train_and_save_model(df,sequence_length=5, epochs = 1)
