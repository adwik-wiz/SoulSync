import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50()

num_emotion_classes = 6  # Change as per your dataset
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),  # First additional layer
    nn.BatchNorm1d(256),                   # Batch Normalization
    nn.ReLU(),                             # Activation function
    nn.Dropout(0.3),                       # Dropout for regularization

    nn.Linear(256, 512),                   # Second additional layer
    nn.BatchNorm1d(512),                   # Batch Normalization
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(512, 256),                   # Third additional layer
    nn.BatchNorm1d(256),                   # Batch Normalization
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(256, num_emotion_classes)    # Final classification layer
)

# Move the model to the appropriate device
model = model.to(device)

checkpoint = torch.load("resnet50_finetune_64.pth", map_location=torch.device('cpu'))

model.load_state_dict(torch.load(r"D:\SoulSync\resnet50_finetune_64.pth", map_location=device, weights_only=True))

transform = transforms.Compose([
    transforms.Resize((72, 72)),  # Resize to 72x72
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

image_path =  r"D:\SoulSync\test_images\test_image_1.jpg"

image = Image.open(image_path).convert("RGB")
image.show()

image_resized = image.resize((72, 72))
image = transform(image).unsqueeze(0)  # Add batch dimension

model.eval()

# Get the last layer output
with torch.no_grad():
    output = model(image)
probabilities = F.softmax(output, dim=1)
print(probabilities)

valence = probabilities[0][0]*0.45 + probabilities[0][1]*0.3 + probabilities[0][2]*0.9 + probabilities[0][3]*0.77 + probabilities[0][4]*-0.82 + probabilities[0][5]*-0.4
arousal = probabilities[0][0]*0.2 + probabilities[0][1]*-0.98 + probabilities[0][2]*-0.56 + probabilities[0][3]*0.72 + probabilities[0][4]*-0.4 + probabilities[0][5]*0.8
valence = (valence+1)/2
arousal = (arousal+1)/2
valence = valence.item()
arousal = arousal.item()
print(valence, arousal)

df = pd.read_json("top_881_track_metadata.json")

distance = np.sqrt((df['valence'] - valence) ** 2 + (df['arousal'] - arousal) ** 2)

# Find the closest point
closest_point = df.loc[distance.idxmin()]

# Display the closest point
print("Closest Point:")
print(closest_point)