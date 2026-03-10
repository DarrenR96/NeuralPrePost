import tomllib

def load_toml_file(file_path: str):
    with open(file_path, 'rb') as f:
        loaded_data = tomllib.load(f)
    return loaded_data