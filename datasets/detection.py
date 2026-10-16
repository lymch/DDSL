from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, split, resize, imagesize, load_images=True):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.load_images = load_images
        self.imagesize = (3, imagesize, imagesize)
        folder = self.root / category / split
        if not folder.is_dir():
            raise FileNotFoundError("Dataset split not found: " + str(folder))
        self.samples = []
        for defect_dir in sorted(folder.iterdir()):
            if not defect_dir.is_dir():
                continue
            for path in sorted(defect_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    if split in {"train", "validation"} and defect_dir.name != "good":
                        raise ValueError("Training/calibration must be normal-only: " + str(path))
                    self.samples.append((path, defect_dir.name))
        if not self.samples:
            raise ValueError("Dataset split contains no images: " + str(folder))
        self.transform = None
        if load_images:
            self.transform = transforms.Compose([
                transforms.Resize([resize, resize]),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, defect = self.samples[index]
        sample = {
            "is_anomaly": int(defect != "good"),
            "image_name": path.relative_to(self.root).as_posix(),
            "image_path": str(path),
        }
        if self.load_images:
            with Image.open(path) as image:
                sample["image"] = self.transform(image.convert("RGB"))
        return sample


def make_loader(config, category, split, load_images=True):
    dataset = DetectionDataset(
        config["data_path"], category, split,
        config["resize"], config["imagesize"], load_images=load_images,
    )
    workers = config.get("num_workers", 2)
    kwargs = dict(
        batch_size=config["batch_size"], shuffle=split == "train",
        num_workers=workers, pin_memory=load_images and torch.cuda.is_available(),
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(dataset, **kwargs)
