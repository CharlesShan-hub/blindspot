DEFAULT_BASE_PATH = ""
BASE_PATH = DEFAULT_BASE_PATH

def check_base_path():
    if BASE_PATH == DEFAULT_BASE_PATH:
        raise ValueError("BASE_PATH is not set. Please call set_base_path() before using the library.")

def set_base_path(path):
    global BASE_PATH
    BASE_PATH = path
    import importlib
    pkg = importlib.import_module(__name__.rsplit('.', 1)[0])
    setattr(pkg, 'BASE_PATH', path)