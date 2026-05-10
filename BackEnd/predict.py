import sys
import json
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image


class TransferModel(nn.Module):
    def __init__(self, num_classes, base_model):
        super().__init__()
        self.base_model = base_model
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.base_model(x))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('model_config.json') as f:
    config = json.load(f)
NUM_CLASSES = config['num_classes']

with open('class_to_idx.json') as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

backbone = resnet50(weights=None)
backbone.fc = nn.Identity()
model = TransferModel(NUM_CLASSES, backbone)
model.load_state_dict(torch.load('tfmodel.pth', map_location=device))
model.to(device)
model.eval()

print(f'Model loaded. Classes: {list(class_to_idx.keys())}')


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def predict(image_path: str) -> dict:
    """
    Given a path to an image file, return the predicted class and confidence scores.

    Returns:
        {
            'predicted_class': 'SUV',
            'confidence': 0.923,
            'all_scores': {'SUV': 0.923, 'Sedan': 0.045, ...}
        }
    """
    image = Image.open(image_path)
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    scores = {idx_to_class[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)}
    best_idx = probs.argmax().item()

    return {
        'predicted_class': idx_to_class[best_idx],
        'confidence': round(probs[best_idx].item(), 4),
        'all_scores': dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)),
    }


# if __name__ == '__main__':
#     if len(sys.argv) < 2:
#         print('Usage: python predict.py <image_path>')
#         sys.exit(1)

#     result = predict(sys.argv[1])
#     print(f"\nPredicted class : {result['predicted_class']}")
#     print(f"Confidence      : {result['confidence']*100:.1f}%")
#     print(f"\nAll scores:")
#     for cls, score in result['all_scores'].items():
#         bar = '█' * int(score * 30)
#         print(f"  {cls:<15} {score*100:5.1f}%  {bar}")
