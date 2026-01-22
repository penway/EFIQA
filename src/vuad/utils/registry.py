import torch

class BaseRegistry:
    _registry = {}
    ITEM_TYPE = "item_type"
    
    @classmethod
    def register(
        cls,
        name: str | None = None,
        *,
        overwrite: bool = False,
    ):

        def decorator(obj):
            key = (name or obj.__name__).lower()
            if not overwrite and key in cls._registry:
                raise KeyError(f"{cls.ITEM_TYPE} '{key}' already registered.")
            cls._registry[key] = obj
            return obj

        return decorator

    @classmethod
    def get(cls, name: str, *args, **kwargs):
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"{cls.ITEM_TYPE} '{name}' not in registry.")
        target = cls._registry[key]
        return target(*args, **kwargs)

    @classmethod
    def has(cls, name: str) -> bool:
        return name.lower() in cls._registry

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())

class ModelRegistry(BaseRegistry):
    ITEM_TYPE = "Model"

class LossRegistry(BaseRegistry):
    ITEM_TYPE = "Loss"

class DegradeRegistry(BaseRegistry):
    ITEM_TYPE = "Degrade"

class OptimRegistry(BaseRegistry):
    ITEM_TYPE = "Optim"

OptimRegistry._registry = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}
