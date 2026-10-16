"""The WideResNet50 backbone used by DDSL."""
import torchvision.models as models


def load(name):
    if name != "wideresnet50":
        raise ValueError("This method implementation supports backbone 'wideresnet50' only.")
    return models.wide_resnet50_2(pretrained=True)
