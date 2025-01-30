from src.pricemodel.analysis import *

model_p = 'outputs/models/property_model_20250129_234546.pth'
proc_p = 'outputs/models/property_model_20250129_234546_processor.pkl'

analyze_saved_model_features(model_p,proc_p)