from .aodnet import AODNet
from .ffanet import FFANet
from .transformer import DehazeTransformer
from .mamba import MambaDehazeNet

def get_model(model_name: str):
    models = {
        "aodnet": AODNet,
        "ffanet": FFANet,
        "transformer": DehazeTransformer,
        "mamba": MambaDehazeNet
    }
    name = model_name.lower()
    if name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models.keys())}")
    return models[name]()