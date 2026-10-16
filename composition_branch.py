"""Composition branch: component histograms, positional features and discriminator."""
import logging
import os

import numpy as np
import torch
import tqdm

from composition_features import (
    PRODUCTS, get_logic_hist_features, get_logic_pos_features, read_seg_maps,
)
from utils import TBWrapper

LOGGER = logging.getLogger(__name__)


def init_weight(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)


class CompositionDiscriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=2, hidden=None):
        super().__init__()
        width = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            input_width = in_planes if i == 0 else width
            width = int(width // 1.5) if hidden is None else hidden
            self.body.add_module("block%d" % (i + 1), torch.nn.Sequential(
                torch.nn.Linear(input_width, width),
                torch.nn.BatchNorm1d(width),
                torch.nn.LeakyReLU(0.2),
            ))
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(width, 1, bias=False), torch.nn.Sigmoid()
        )
        self.apply(init_weight)

    def forward(self, features):
        return self.tail(self.body(features))


class CompositionBranch(torch.nn.Module):
    def __init__(self, device, category, seg_path, meta_epochs=160,
                 logic_dsc_layers=2, logic_dsc_hidden=256, logic_dsc_lr=0.0002,
                 logic_noise=0.001):
        super().__init__()
        if category not in PRODUCTS:
            raise ValueError("No component feature definition for " + category)
        if not seg_path:
            raise ValueError("dataset.seg_path is required for the composition branch.")
        self.device = device
        self.category = category
        self.seg_path = seg_path
        self.mask_cls_num, self.logic_feats_dim, self.use_pos = PRODUCTS[category]
        self.meta_epochs = meta_epochs
        self.logic_noise = logic_noise
        self.logic_dsc = CompositionDiscriminator(
            self.logic_feats_dim, logic_dsc_layers, logic_dsc_hidden
        ).to(device)
        self.logic_dsc_opt = torch.optim.Adam(
            self.logic_dsc.parameters(), lr=logic_dsc_lr, weight_decay=1e-5
        )

    def set_model_dir(self, model_dir, dataset_name):
        self.ckpt_dir = os.path.join(model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def extract_features(self, image_names):
        masks = torch.from_numpy(read_seg_maps(self.seg_path, image_names)).to(self.device)
        features = get_logic_hist_features(masks, self.mask_cls_num).reshape(len(image_names), -1)
        if self.use_pos:
            positions = torch.tensor(
                get_logic_pos_features(masks, self.category),
                dtype=torch.float32, device=self.device,
            )
            features = torch.cat((features, positions), dim=1)
        if features.shape[1] != self.logic_feats_dim:
            raise ValueError("Component feature dimension does not match the discriminator.")
        return features

    def trainer(self, training_data):
        if self.meta_epochs < 1:
            raise ValueError("meta_epochs must be positive.")
        path = os.path.join(self.ckpt_dir, "ckpt_last.pth")
        if os.path.exists(path):
            raise FileExistsError("Choose a new run_name; checkpoint already exists: " + path)
        self.logger = TBWrapper(os.path.join(self.ckpt_dir, "tb"))
        try:
            for epoch in range(self.meta_epochs):
                loss = self._train_discriminator(training_data)
                LOGGER.info("Composition epoch %d/%d | loss=%.6f",
                            epoch + 1, self.meta_epochs, loss)
            torch.save({
                "epoch": self.meta_epochs,
                "selection": "last",
                "random_seeds": getattr(self, "random_seeds", None),
                "category": self.category,
                "logic_dsc": self.logic_dsc.state_dict(),
            }, path)
        finally:
            self.logger.logger.close()
        return path

    def _train_discriminator(self, training_data):
        self.logic_dsc.train()
        losses = []
        for batch in tqdm.tqdm(training_data, desc="Composition training", leave=False):
            self.logic_dsc_opt.zero_grad()
            features = self.extract_features(batch["image_name"])
            anomalous = features.clone()
            for i in range(len(features)):
                anomalous[i] = anomalous[i] + torch.randn_like(anomalous[i]) * self.logic_noise
            scores = self.logic_dsc(torch.cat((features, anomalous), dim=0))
            normal, abnormal = scores[:len(features)], scores[len(features):]
            loss = (
                torch.nn.functional.binary_cross_entropy(normal, torch.zeros_like(normal))
                + torch.nn.functional.binary_cross_entropy(abnormal, torch.ones_like(abnormal))
            )
            loss.backward()
            self.logic_dsc_opt.step()
            value = loss.detach().cpu().item()
            self.logger.logger.add_scalar("loss", value, self.logger.g_iter)
            self.logger.step()
            losses.append(value)
        return float(np.mean(losses))

    def load_last_checkpoint(self):
        path = os.path.join(self.ckpt_dir, "ckpt_last.pth")
        if not os.path.isfile(path):
            raise FileNotFoundError("Train the composition branch first: " + path)
        state = torch.load(path, map_location=self.device)
        if state.get("selection") != "last":
            raise ValueError("Expected a final-epoch checkpoint: " + path)
        if state.get("category", self.category) != self.category:
            raise ValueError("Checkpoint category does not match the configured product.")
        self.logic_dsc.load_state_dict(state["logic_dsc"])

    def predict(self, dataloader):
        self.logic_dsc.eval()
        scores, labels, paths = [], [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Composition inference", leave=False):
                features = self.extract_features(batch["image_name"])
                values = self.logic_dsc(features).reshape(-1).cpu().numpy()
                scores.extend(values)
                labels.extend(batch["is_anomaly"].tolist())
                paths.extend(batch["image_path"])
        return np.asarray(scores), np.asarray(labels), paths
