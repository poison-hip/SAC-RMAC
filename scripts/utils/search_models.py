from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(filter="reinforcement-learning", search="sac panda")

print("Found Panda models:")
for model in models:
    if "PickAndPlace" not in model.modelId:
        print(model.modelId)

print("\nFound Fetch models:")
models_fetch = api.list_models(filter="reinforcement-learning", search="sac fetch")
for model in models_fetch:
    if "PickAndPlace" not in model.modelId:
        print(model.modelId)
