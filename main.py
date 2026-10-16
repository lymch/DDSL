"""DDSL training and detection: appearance, composition or calibrated max fusion."""
import argparse
import json
import logging
import os

import numpy as np
import yaml

import backbones
from appearance_branch import AppearanceBranch
from datasets.detection import make_loader
from composition_branch import CompositionBranch
from fusion import align_scores, fit_normalization, fuse_predictions
from metrics import detection_aurocs
from utils import create_storage_folder, fix_seeds, set_torch_device

LOGGER = logging.getLogger(__name__)


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def branch_seeds(config, fallback):
    """Resolve independent initialization/training seeds for one branch."""
    base = config.get("seed", fallback)
    seeds = {name: config.get(name, base) for name in ("init_seed", "train_seed")}
    for name, value in seeds.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
            raise ValueError(name + " must be an integer in [0, 2**32).")
    return seeds


def build_appearance(config, device):
    options = dict(config["net"])
    names = options.pop("backbone_names")
    if len(names) != 1:
        raise ValueError("Configure one appearance backbone; ensembles are not implemented.")
    backbone = backbones.load(names[0])
    model = AppearanceBranch(device)
    image_size = config["dataset"]["imagesize"]
    model.load(backbone=backbone, device=device, input_shape=(3, image_size, image_size), **options)
    return model


def build_composition(config, category, device):
    return CompositionBranch(
        device=device, category=category,
        seg_path=config["dataset"]["seg_path"], **config["focnet"]
    )


def configure_model_path(model, config, category):
    root = create_storage_folder(
        config["results_path"], config["log_project"], config["log_group"], config["run_name"]
    )
    name = config["dataset"]["name"] + "_" + category
    model.set_model_dir(os.path.join(root, "models", "0"), name)
    return root


def predict_split(model, config, category, split, composition=False):
    loader = make_loader(config["dataset"], category, split, load_images=not composition)
    return model.predict(loader)


def main(config, logic_config):
    mode, branch = config["mode"], config["branch_type"]
    if branch not in {"struct", "logic", "add"} or mode not in {"train", "test"}:
        raise ValueError("Unsupported execution mode or branch.")
    active = logic_config if branch == "logic" else config
    if active["dataset"]["name"] not in {"mvtec", "visa", "mvtec_loco"}:
        raise ValueError("Unsupported dataset layout.")
    if branch in {"logic", "add"} and logic_config["dataset"]["name"] != "mvtec_loco":
        raise ValueError("Composition requires the LOCO/PCBA dataset layout.")
    if branch == "add":
        if config["dataset"]["name"] != "mvtec_loco":
            raise ValueError("Dual-branch detection requires the LOCO/PCBA layout.")
        if config["dataset"]["subdatasets"] != logic_config["dataset"]["subdatasets"]:
            raise ValueError("Both branches must use identical categories.")
    categories = active["dataset"]["subdatasets"]
    if not categories:
        raise ValueError("Configure at least one category.")
    device = set_torch_device(config["gpu"])
    results = []
    for category in categories:
        appearance, composition = None, None
        if branch != "logic":
            seeds = branch_seeds(config, config["seed"])
            fix_seeds(seeds["init_seed"])
            appearance = build_appearance(config, device)
            appearance.random_seeds = seeds
        if branch != "struct":
            seeds = branch_seeds(logic_config, config["seed"])
            fix_seeds(seeds["init_seed"])
            composition = build_composition(logic_config, category, device)
            composition.random_seeds = seeds
        models = []
        if appearance is not None:
            root = configure_model_path(appearance, config, category)
            models.append((appearance, config, False))
        if composition is not None:
            logic_root = configure_model_path(composition, logic_config, category)
            models.append((composition, logic_config, True))
        result_root = logic_root if branch == "logic" else root
        LOGGER.info("%s | category=%s | branch=%s", mode, category, branch)
        for model, _, is_composition in models:
            LOGGER.info("%s seeds: %s", "composition" if is_composition else "appearance",
                        model.random_seeds)
        if mode == "train":
            # Check both destinations before starting either branch.
            for model, _, _ in models:
                path = os.path.join(model.ckpt_dir, "ckpt_last.pth")
                if os.path.exists(path):
                    raise FileExistsError("Use a new run name; final checkpoint exists: " + path)
            for model, model_config, is_composition in models:
                # Reset before loader creation and training, independently of other branches.
                fix_seeds(model.random_seeds["train_seed"])
                loader = make_loader(
                    model_config["dataset"], category, "train", load_images=not is_composition
                )
                model.trainer(loader)
            continue
        for model, _, _ in models:
            model.load_last_checkpoint()
        if branch == "add":
            appearance_val = predict_split(appearance, config, category, "validation")
            composition_val = predict_split(
                composition, logic_config, category, "validation", composition=True
            )
            # Calibration uses the same normal images, regardless of loader ordering.
            align_scores(appearance_val[0], appearance_val[2], composition_val[2])
            appearance_stats = fit_normalization(appearance_val[0], appearance_val[1])
            composition_stats = fit_normalization(composition_val[0], composition_val[1])
            predictions = fuse_predictions(
                predict_split(appearance, config, category, "test"),
                predict_split(composition, logic_config, category, "test", composition=True),
                appearance_stats, composition_stats,
            )
        else:
            model, model_config, is_composition = models[0]
            predictions = predict_split(
                model, model_config, category, "test", composition=is_composition
            )
        row = {"dataset_name": active["dataset"]["name"] + "_" + category}
        row.update(detection_aurocs(*predictions, logical=active["dataset"]["name"] == "mvtec_loco"))
        results.append(row)
        LOGGER.info("Detection results: %s", row)

    if mode == "train":
        return []
    if len(results) > 1:
        results.append({
            "dataset_name": "mean",
            **{key: float(np.mean([row[key] for row in results]))
               for key in results[0] if key != "dataset_name"},
        })
    suffix = categories[0] if len(categories) == 1 else "all"
    path = os.path.join(result_root, "{}_results_{}.json".format(branch, suffix))
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    LOGGER.info("Saved detection results to %s", path)
    return results


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mvtec_struct.yaml")
    parser.add_argument("--logic-config", default="configs/mvtec_loco.yaml")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--branch-type", choices=["struct", "logic", "add"], default="add")
    parser.add_argument("--category", default=None, help="Override configured categories.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    logic_config = load_yaml(args.logic_config) if args.branch_type != "struct" else None
    for current in (config, logic_config):
        if current is not None:
            if args.seed is not None:
                current.update(seed=args.seed, init_seed=args.seed, train_seed=args.seed)
            if args.run_name:
                current["run_name"] = args.run_name
            if args.category:
                current["dataset"]["subdatasets"] = [args.category]
    config.update(mode=args.mode, branch_type=args.branch_type)
    main(config, logic_config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
