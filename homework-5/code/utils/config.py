import yaml
from easydict import EasyDict as ezdict

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = ezdict(yaml.load(file, Loader=yaml.SafeLoader))
    return config
