# IQX/src/vuad/dataset/__init__.py
import pkgutil
import importlib

for _, module_name, _ in pkgutil.walk_packages(__path__):
    importlib.import_module(f"{__name__}.{module_name}")
