import json

from rootnav2.loader.root_loader import rootLoader

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
       "root": rootLoader,

    }[name]


def get_data_path(name, config_file="config.json"):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]["data_path"]
