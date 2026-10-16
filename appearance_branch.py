"""Local appearance branch with radial-constrained anomaly synthesis (RCAS)."""
import logging
import os

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import tqdm

import Struct.common as common
from Struct.model import Discriminator, Projection, PatchMaker
from rcas import recompose_centers, synthesize_anomalies
from utils import TBWrapper

LOGGER = logging.getLogger(__name__)


class AppearanceBranch(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=100,
            dsc_layers=3,
            pre_proj=1,
            noise=0.015,
            clamp_min=0.1,
            clamp_max=3.0,
            lr=0.0001,
            limit=-1,
    ):
        # 1.Feature Extractor
        self.backbone = backbone.to(device)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, False
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs

        # 2.Feature Adaptor
        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr)  # Feature Adaptor Optimizer

        # 3.Discriminator
        self.dsc_layers = dsc_layers
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr * 2, weight_decay=1e-5)  # Discriminator Optimizer

        self.c1 = torch.tensor([]).to(device)
        self.noise = noise
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.limit = limit

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.model_dir = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir  # /results
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)  # results/models
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")  # results/models/tb
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = None

    def _embed(self, images):
        """Returns feature embeddings for images."""
        self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(patch_features)):
            feature = patch_features[i]
            patch_dims = patch_shapes[i]

            feature = feature.reshape(
                feature.shape[0], patch_dims[0], patch_dims[1], *feature.shape[2:]
            )
            feature = feature.permute(0, 3, 4, 5, 1, 2)
            perm_base_shape = feature.shape
            feature = feature.reshape(-1, *feature.shape[-2:])
            feature = F.interpolate(
                feature.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            feature = feature.squeeze(1)
            feature = feature.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            feature = feature.permute(0, 4, 5, 1, 2, 3)
            feature = feature.reshape(len(feature), -1, *feature.shape[-3:])
            patch_features[i] = feature

        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features, patch_shapes

    def _initialize_center(self, train_data):
        """Average adapted features over samples, including a partial last batch."""
        center_sum = None
        sample_count = 0
        with torch.no_grad():
            for data in train_data:
                img = data["image"].to(torch.float).to(self.device)
                outputs = self._embed(img)[0]
                if self.pre_proj > 0:
                    outputs = self.pre_projection(outputs)
                outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1])
                batch_sum = outputs.sum(dim=0)
                center_sum = batch_sum if center_sum is None else center_sum + batch_sum
                sample_count += img.shape[0]
            if sample_count == 0:
                raise ValueError("Cannot initialize the center from an empty training loader.")
            self.c1 = center_sum / sample_count

    def trainer(self, train_data):
        if self.meta_epochs < 1:
            raise ValueError("meta_epochs must be positive.")
        checkpoint_path = os.path.join(self.ckpt_dir, "ckpt_last.pth")
        if os.path.exists(checkpoint_path):
            raise FileExistsError(
                "Final checkpoint already exists. Choose a new run_name: " + checkpoint_path
            )
        self._initialize_center(train_data)

        self.logger = TBWrapper(self.tb_dir)
        pbar = tqdm.tqdm(range(self.meta_epochs), unit="epoch")
        for i_epoch in pbar:
            self._train_discriminator(train_data, i_epoch, pbar)
        state = {
            "epoch": self.meta_epochs,
            "selection": "last",
            "random_seeds": getattr(self, "random_seeds", None),
            "discriminator": self.discriminator.state_dict(),
            "forward_modules": self.forward_modules.state_dict(),
            "center": self.c1,
        }
        if self.pre_proj > 0:
            state["pre_projection"] = self.pre_projection.state_dict()
        torch.save(state, checkpoint_path)
        self.logger.logger.close()
        LOGGER.info("Saved final epoch %d to %s", self.meta_epochs, checkpoint_path)
        return checkpoint_path

    def load_last_checkpoint(self):
        checkpoint_path = os.path.join(self.ckpt_dir, "ckpt_last.pth")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Final checkpoint missing; train this branch first: " + checkpoint_path
            )
        state = torch.load(checkpoint_path, map_location=self.device)
        if state.get("selection") != "last":
            raise ValueError("Expected a final-epoch checkpoint: " + checkpoint_path)
        self.discriminator.load_state_dict(state["discriminator"])
        self.forward_modules.load_state_dict(state["forward_modules"])
        if self.pre_proj > 0:
            self.pre_projection.load_state_dict(state["pre_projection"])
        self.c1 = state["center"].to(self.device)

    def _train_discriminator(self, train_data, cur_epoch, pbar):
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss = []
        sample_num = 0
        for data_item in train_data:
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()
            self.dsc_opt.zero_grad()

            # 1.Feature Extract and Adapt
            img = data_item["image"]  # (N, C, H, W)
            img = img.to(torch.float).to(self.device)
            if self.pre_proj > 0:
                true_feats = self.pre_projection(self._embed(img)[0])
            else:
                true_feats = self._embed(img)[0]

            center_feats = recompose_centers(true_feats, self.c1, img.shape[0])
            fake_feats = synthesize_anomalies(
                true_feats, center_feats, self.noise, self.clamp_min, self.clamp_max
            )
            trues, fakes = true_feats, fake_feats

            scores = self.discriminator(torch.concat([trues, fakes]))
            true_scores = scores[:len(trues)]
            fake_scores = scores[len(trues):]
            true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
            fake_loss = torch.nn.BCELoss()(fake_scores, torch.ones_like(fake_scores))
            loss = true_loss + fake_loss

            loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()
            self.dsc_opt.step()

            pix_true = true_scores.detach()
            pix_fake = fake_scores.detach()
            p_t = (pix_true < 0.5).sum() / pix_true.shape[0]
            p_f = (pix_fake >= 0.5).sum() / pix_fake.shape[0]

            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            self.logger.logger.add_scalar("p_true", p_t, self.logger.g_iter)
            self.logger.logger.add_scalar("p_fake", p_f, self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())

            all_loss_ = np.mean(all_loss)
            sample_num += img.shape[0]

            pbar_str = f"epoch:{cur_epoch}"
            pbar_str += f" loss:{all_loss_:.2e}"
            pbar_str += f" sample:{sample_num}"
            pbar.set_description_str(pbar_str)

            if sample_num > self.limit != -1:
                break

    def predict(self, dataloader):
        """Return only image-level scores, labels and paths."""
        scores, labels, paths = [], [], []
        for batch in tqdm.tqdm(dataloader, desc="Appearance inference", leave=False):
            scores.extend(self._predict(batch["image"]))
            labels.extend(batch["is_anomaly"].tolist())
            paths.extend(batch["image_path"])
        return np.asarray(scores), np.asarray(labels), paths

    def _predict(self, images):
        images = images.to(torch.float).to(self.device)
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            features, patch_shapes = self._embed(images)
            if self.pre_proj > 0:
                features = self.pre_projection(features)
            scores = self.discriminator(features)
            scores = self.patch_maker.unpatch_scores(scores, batchsize=images.shape[0])
            patch_scores = scores.reshape(images.shape[0], *patch_shapes[0])
            scores = self._image_scores(patch_scores, images.shape[-2:])
        return scores

    def _image_scores(self, patch_scores, output_size):
        """Upsample, smooth each image independently, then take its maximum."""
        maps = F.interpolate(
            patch_scores.unsqueeze(1), size=output_size,
            mode="bilinear", align_corners=False,
        ).squeeze(1).detach().cpu().numpy()
        # Match the original anomaly-map smoothing (sigma in input-image pixels).
        return np.asarray([
            gaussian_filter(score_map, sigma=4, mode="reflect").max()
            for score_map in maps
        ])
