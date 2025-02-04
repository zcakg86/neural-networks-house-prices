import panel as pn

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pricemodel')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from analysis import *

model = '20250131_172455'
model_p = f'outputs/models/property_model_{model}.pth'
proc_p = f'outputs/models/property_model_{model}_processor.pkl'

pn.extension('ipywidgets')
pn.panel("Hi").servable()
pn.panel('Update').servable()

analyze_saved_model_features(model_p,proc_p)
    




