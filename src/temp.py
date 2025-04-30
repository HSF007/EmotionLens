import shutil
import os
import json
f = open('metrics/evaluation_metrics.json')
m = json.load(f)
f.close()
if m['accuracy'] > 0.60:
    shutil.copy('models/model.h5', 'models/best_model/model.h5')