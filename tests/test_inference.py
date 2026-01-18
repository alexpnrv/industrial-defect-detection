import torch
import pytest
from src.models.resnet import ResNetClassifier
from src.inference.predictor import Predictor
from PIL import Image

@pytest.fixture
def dummy_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (224, 224), color=(255, 0, 0)).save(img_path)
    return str(img_path)

@pytest.fixture
def model():
    return ResNetClassifier(num_classes=2)

def test_predictor_runs(model, dummy_image):
    device = torch.device("cpu")
    predictor = Predictor(model, device)
    label = predictor.predict(dummy_image)
    assert label in [0, 1]