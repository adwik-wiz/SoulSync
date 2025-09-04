from fastapi import FastAPI, File, UploadFile, HTTPException, status
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
import numpy as np
import pandas as pd


async def validate_file(file: UploadFile, max_size: int = None, mime_types: list = None):
    """
    Validate a file by checking the size and mime types a.k.a file types
    """
    if mime_types and file.content_type not in mime_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You can only upload image files"
        )

    file.file.seek(0, 2)  # Move to the end of the file
    file_size = file.file.tell()  # Get the file size
    file.file.seek(0)  # Move back to the beginning of the file

    if max_size and file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size is too large"
        )

    return file


app = FastAPI()

# Load your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50()

num_emotion_classes = 6
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_emotion_classes)
)

model.load_state_dict(torch.load("resnet50_finetune_64.pth", map_location=device))
model.to(device)
model.eval()

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load metadata
df = pd.read_json("top_881_track_metadata.json")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    print(f"Content type: {file.content_type}")

    valid_types = ["image/png", "image/jpeg", "image/bmp"]
    max_size = 5 * 1024 * 1024  # 5 MB
    await validate_file(file, max_size, valid_types)

    contents = await file.read()
    print(f"File size: {len(contents)} bytes")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("Image opened successfully")
    except Exception as e:
        print(f"Error opening image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
    probabilities = torch.nn.functional.softmax(output, dim=1)

    valence = probabilities[0][0] * 0.45 + probabilities[0][1] * 0.3 + probabilities[0][2] * 0.9 + probabilities[0][
        3] * 0.77 + probabilities[0][4] * -0.82 + probabilities[0][5] * -0.4
    arousal = probabilities[0][0] * 0.2 + probabilities[0][1] * -0.98 + probabilities[0][2] * -0.56 + probabilities[0][
        3] * 0.72 + probabilities[0][4] * -0.4 + probabilities[0][5] * 0.8
    valence = (valence + 1) / 2
    arousal = (arousal + 1) / 2
    valence = valence.item()
    arousal = arousal.item()

    distance = np.sqrt((df['valence'] - valence) ** 2 + (df['arousal'] - arousal) ** 2)
    closest_point = df.loc[distance.idxmin()]

    return {
        "valence": valence,
        "arousal": arousal,
        "closest_track": closest_point.to_dict()
    }