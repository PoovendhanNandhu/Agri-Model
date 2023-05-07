import torch
import cv2
import os
import glob
from transformers import ViTFeatureExtractor, ViTForImageClassification

DATASET_DIR = '/path/to/dataset'
ORGANIC_PATH = os.path.join(DATASET_DIR, 'organic')
NON_ORGANIC_PATH = os.path.join(DATASET_DIR, 'non_organic')

IMAGE_SIZE = 224
BATCH_SIZE = 32

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

model.classifier = torch.nn.Linear(model.config.hidden_size, 1)
model.config.label2id = {0: 0, 1: 1}

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

class VegetableDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label):
        self.image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.label = label
        self.image_size = IMAGE_SIZE

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        label = self.label
        return image, label

organic_dataset = VegetableDataset(ORGANIC_PATH, 1)
non_organic_dataset = VegetableDataset(NON_ORGANIC_PATH, 0)

train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([organic_dataset, non_organic_dataset]), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(10):
    train_loss = 0
    train_accuracy = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        features = feature_extractor(data, return_tensors='pt').pixel_values
        output = model(features).squeeze()
        loss = criterion(output, target.float())
        predictions = torch.round(torch.sigmoid(output))
        accuracy = (predictions == target).float().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_accuracy += accuracy.item()

    print(f'Epoch {epoch+1} - Training Loss: {train_loss/train_dataset.__len__():.4f} - Training Accuracy: {train_accuracy/train_dataset.__len__():.4f}') 

torch.save(model.state_dict(), 'vegetable_classifier.pt')
model.load_state_dict(torch.load('vegetable_classifier.pt'))

def predict(image_path, model, feature_extractor):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    features = feature_extractor(image.unsqueeze(0), return_tensors='pt').pixel_values
    output = model(features).squeeze()
    prediction = torch.sigmoid(output).item()
    if prediction > 0.5:
        print('Organic')
    else:
        print('Non-organic')

image_path = '/path/to/image'
predict(image_path, model, feature_extractor)
