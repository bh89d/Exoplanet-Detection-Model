import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

SEQ_LEN = 4096
BATCH_SIZE = 64
EPOCHS = 1000

class LightCurveDataset(Dataset):
  
  def __init__(self, files, labels):
    self.files = files
    self.labels = labels
    
  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, idx):
    
    flux = np.load(self.files[idx])
    
    flux = (flux - np.median(flux))/np.std(flux)
    
    if len(flux) > SEQ_LEN:
      flux = flux[:SEQ_LEN]
    
    else:
      flux = np.pad(flux, (0, SEQ_LEN - len(flux)))
      
    flux = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)
    
    label = torch.tensor(self.labels[idx], dtype=torch.float32)
    
    return flux, label

class ExoplanetCNN(torch.nn):
  
  def __init__(self):
    self.__init__()
    
    self.conv = nn.Sequential(
      nn.Conv1d(1, 16, kernel_size=7, padding=3),
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Conv1d(16, 32, kernel_size=5, padding=2),
      nn.Conv1d(32, 64, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.MaxPool1d(2),
    )
    
    self.fc == nn.Sequential(
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
    
    