# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CSRNet
from dataset import CrowdDataset
import os

# Hyperparams
batch_size = 1
lr = 1e-5
epochs = 400
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)

# Data
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CrowdDataset('processed/part_B/images', 'processed/part_B/density', transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = CSRNet(load_weights=True).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for img, density in train_loader:
        img, density = img.cuda(), density.cuda()
        
        pred = model(img)
        loss = criterion(pred, density)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    if (epoch+1) % 50 == 0:
        torch.save(model.state_dict(), f"{save_dir}/csrnet_epoch{epoch+1}.pth")