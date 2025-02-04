from src.pricemodel.analysis import *
model = '20250131_172455'
model_p = f'outputs/models/property_model_{model}.pth'
proc_p = f'outputs/models/property_model_{model}_processor.pkl'

analyze_saved_model_features(model_p,proc_p)

