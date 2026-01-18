import albumentations as a
from albumentations import ToTensorV2

def get_train_transforms():
    return a.Compose([
        a.HorizontalFlip(p=0.5),
        a.VerticalFlip(p=0.5),
        a.Normalize(),
        ToTensorV2(),
    ])

def get_val_transforms():
    return a.Compose([
        a.Normalize(),
        ToTensorV2(),
    ])