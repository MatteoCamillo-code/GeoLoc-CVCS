from src.utils.io import read_json
import numpy as np

def compute_history_val_acc_from_history(cfg, loader_dict, project_root):
    scene_sizes = {}
    scene_accs = {}

    # collect
    for scene in cfg.scenes:
        history = read_json(project_root / "outputs" / "history" / f"{cfg.model_name}_{scene}_history.json")
        scene_accs[scene] = np.asarray(history["val_acc"], dtype=float)  # (E,)
        scene_sizes[scene] = len(loader_dict[scene]["val_loader"].dataset)

    # choose number of epochs to aggregate (safe if some runs are shorter)
    E = min(a.shape[0] for a in scene_accs.values())

    total_size = sum(scene_sizes.values())
    overall = np.zeros(E, dtype=float)

    for scene in cfg.scenes:
        w = scene_sizes[scene]
        overall += scene_accs[scene][:E] * w

    overall /= total_size
    return overall
    
def overall_val_acc_from_history(cfg, project_root, version: int):
    total_size = 0
    total_acc = 0.0
    output = {}

    # collect
    for scene in cfg.scenes:
        history = read_json(project_root / "outputs" / "history" / f"{cfg.model_name}_{scene}_v{version}.json")
        acc = history["val_acc"][-1]  # last epoch
        output[scene] = acc
        size = history["val_size"]
        total_acc += acc * size
        total_size += size

    total_acc /= total_size
    output["overall"] = total_acc
    return output