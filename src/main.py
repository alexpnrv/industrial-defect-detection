from src.models.resnet import ResNetClassifier
from src.data.dataset import DefectDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.training.trainer import Trainer
from torch.utils.data import DataLoader
import torch

def main():
    train_dataset = DefectDataset('data/raw', annotations=[], transform=get_train_transforms())
    val_dataset = DefectDataset('data/raw', annotations=[], transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(num_classes=2)

    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.train_epoch()
    acc = trainer.validate()
    print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()