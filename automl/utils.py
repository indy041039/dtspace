
import os
import yaml
import json
import joblib

def save_model(model, dir_name, model_name):
    os.makedirs(dir_name, exist_ok=True)
    joblib.dump(model, os.path.join(dir_name, model_name))

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def read_json(json_path):
    with open(json_path, 'rb')as f:
        data = json.load(f)
    return data

def read_yaml(yaml_path):
    try:
        with open(yaml_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        print("File not accessible")  # must do sth (send email?)
        exit()
    return config
