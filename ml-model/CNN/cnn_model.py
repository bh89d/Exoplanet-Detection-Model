import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import time
from tqdm import tqdm

metadata = pd.read_csv("data/ml/features/metadata.csv")

groups = metadata["target_id"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state= 42)

train_idx , test_idx = next(
  gss.split(metadata, metadata["label"], groups=groups)
)

train_df = metadata.iloc[train_idx].reset_index(drop = True)
test_df = metadata.iloc[test_idx].reset_index(drop = True)

groups = train_df["target_id"]

train_idx , validation_idx = next(
  gss_val.split(train_df, train_df["label"], groups=groups )
)

train_split = train_df.iloc[train_idx].reset_index(drop = True)
validation_df = train_df.iloc[validation_idx].reset_index(drop = True)

train_df = train_split

SEQ_LEN = 4096
BATCH_SIZE = 64
EPOCHS = 10

class LightCurveDataset(Dataset):
  
  def __init__(self, dataframe):
    self.data = dataframe
    
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    
    row = self.data.iloc[idx]
    
    with np.load(row["file_path"]) as curve:
      
      flux = curve["flux"]
    
    flux = (flux - np.median(flux))/np.std(flux)
    
    if len(flux) > SEQ_LEN:
      flux = flux[:SEQ_LEN]
    
    else:
      flux = np.pad(flux, (0, SEQ_LEN - len(flux)))
      
    flux = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)
    
    label = torch.tensor(row["label"], dtype=torch.float32)
    
    return flux, label
  
train_dataset = LightCurveDataset(train_df)
validation_dataset = LightCurveDataset(validation_df)
test_dataset = LightCurveDataset(test_df)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle= False)

class ExoplanetCNN(nn.Module):
  
  def __init__(self):
    super().__init__()
    
    self.conv = nn.Sequential(
      nn.Conv1d(1, 16, kernel_size=7, padding=3),
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Conv1d(16, 32, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Conv1d(32, 64, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.MaxPool1d(2),
    )
    
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64*512, 128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(128, 1)
    )
    
  def forward(self, x):
    
    x = self.conv(x)
    x = self.fc(x)
    
    return x.squeeze(1)

exoplanet_cnn_1 = ExoplanetCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exoplanet_cnn_1 = exoplanet_cnn_1.to(device)

loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(exoplanet_cnn_1.parameters(), lr = 0.001)

for epoch in range(EPOCHS):
  epoch_start = time.perf_counter()
  
  #Training
  exoplanet_cnn_1.train()
  
  train_loss = 0.0
  
  train_start = time.perf_counter()
  
  train_bar = tqdm(
    train_dataloader,
    desc = f"Epoch {epoch + 1}/{EPOCHS} [Train]"
  )
  
  for flux, labels in train_bar:
    
    flux = flux.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    
    outputs = exoplanet_cnn_1(flux)
    
    batch_loss = loss(outputs, labels)
    
    batch_loss.backward()
    
    optimizer.step()
    
    train_loss += batch_loss.item()
    
    train_bar.set_postfix(
      loss = f"{batch_loss.item():.4f}"
    )
    
  train_loss /= len(train_dataloader)
  
  train_time = time.perf_counter() - train_start

  #Validation
  
  exoplanet_cnn_1.eval()
  
  validation_loss = 0.0
  
  validation_start = time.perf_counter()
  
  validation_bar = tqdm(
    validation_dataloader,
    desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"
  )
  
  with torch.no_grad():
    
    for flux, labels in validation_bar:
      
      flux = flux.to(device)
      labels = labels.to(device)
      
      outputs = exoplanet_cnn_1(flux)
      
      batch_loss = loss(outputs, labels)
      
      validation_loss += batch_loss.item()
      
      validation_bar.set_postfix(
        loss = f"{batch_loss.item():.4f}"
      )
      
  validation_loss /= len(validation_dataloader)
  
  validation_time = time.perf_counter() - validation_start
  
  epoch_time = time.perf_counter() - validation_start
  
  #epoch timing
  
  epoch_time = time.perf_counter() - epoch_start
  
  print(f"\nEpoch {epoch+1}/{EPOCHS}")
  
  print(f"Train Loss : {train_loss:.4f}")
  
  print(f"Validation Loss : {validation_loss:.4f}")
  
  print(f"Train Time : {train_time:.2f}")
  
  print(f"Validation Time : {validation_time:.2f}")
  
  print(f"Total Epoch Time : {epoch_time:.2f}")
  
  
print("/n Starting Test")

exoplanet_cnn_1.eval()

test_loss = 0.0

test_start = time.perf_counter()

test_bar = tqdm(test_dataloader, desc = "Testing")

with torch.no_grad():
  
  for flux, labels in test_bar:
    
    flux = flux.to(device)
    
    labels = labels.to(device)
    
    outputs = exoplanet_cnn_1(flux)
    
    batch_loss = loss(outputs, labels)
    
    test_loss += batch_loss.item()
    
    test_bar.set_postfix(
      loss = f"{batch_loss.item():.4f}"
    )
    
test_loss /= len(test_dataloader)

test_time = time.perf_counter() - test_start

print("\n Test Results:")

print(f"Test Loss : {test_loss:.4f}")

print(f"Test Time : {test_time:.4f}")

print(device)