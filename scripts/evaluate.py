from src.training.trainer import Trainer
from src.models.resnet import ResNetClassifier
from src.data.dataset import DefectDataset
from src.data.transforms import get_val_transforms
from torch.utils.data import DataLoader
import torch

def evaluate():
    val_dataset = DefectDataset('data/raw', annotations=[], transform=get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetClassifier(num_classes=2).to(device)

    trainer = Trainer(model, None, val_loader, device)
    acc = trainer.validate()
    print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate()
