import tomllib
import torch 
import os 

def load_toml_file(file_path: str):
    with open(file_path, 'rb') as f:
        loaded_data = tomllib.load(f)
    return loaded_data

def load_torch_model(torch_model_path: str, model_class):
    model_file = os.path.join(torch_model_path, 'model.pt')
    toml_file = os.path.join(torch_model_path, 'config.toml')
    model_args = load_toml_file(toml_file)
    model = model_class(**model_args)
    model.load_state_dict(torch.load(model_file))
    return model

