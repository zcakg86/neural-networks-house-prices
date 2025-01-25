import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import json
from src.pricemodel import *

df = pd.read_csv('data/sales_2021_on_geo.csv').sample(n=500, random_state = 92)
manager, df_with_predictions = train_and_save_model(df,sequence_length=5, epochs = 20)
