from src.models.resnet import ResNetClassifier
from src.inference.predictor import Predictor
import torch

def infer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(num_classes=2).to(device)

    predictor = Predictor(model, device)
    image_path = "data/raw/sample.jpg"
    label = predictor.predict(image_path)
    print(f"Predicted label: {label}")

if __name__ == "__main__":
    infer()