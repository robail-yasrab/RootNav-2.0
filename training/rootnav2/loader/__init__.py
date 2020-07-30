from rootnav2.loader.roots_loader import rootsLoader

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "roots": rootsLoader,
 
    }[name]


