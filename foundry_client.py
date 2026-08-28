# foundry_client.py — Single shared FoundryLocalManager instance

from foundry_local_sdk import FoundryLocalManager, Configuration
from foundry_local_sdk.exception import FoundryLocalException

try:
    FoundryLocalManager.initialize(Configuration(app_name="study-rag"))
except FoundryLocalException:
    pass  # already initialised by an earlier import

manager = FoundryLocalManager.instance

def get_ready_model(alias: str):
    """Look up a model by alias, downloading and loading it if needed."""
    model = manager.catalog.get_model(alias)
    if model is None:
        raise FoundryLocalException(f"Model '{alias}' not found in catalog")
    if not model.is_cached:
        print(f"[foundry_client] Downloading '{alias}'...")
        model.download()
    if not model.is_loaded:
        print(f"[foundry_client] Loading '{alias}'...")
        model.load()
    return model
