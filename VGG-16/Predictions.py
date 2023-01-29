#%%
import torch
from torchvision import datasets, io, models, ops, transforms, utils
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import optuna
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


from sklearn.metrics import accuracy_score,classification_report

from tqdm import tqdm_notebook as tqdm
import time
import warnings
warnings.simplefilter("ignore")

IMAGE_SIZE=(128,128)
batch_size= 20
learning_rate = 0.003861725786961588
epochs=20

num_classes=4

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(device)

# print(torch.cuda.get_device_name(0))

tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

trainset = datasets.ImageFolder(root='Training', transform=tf)
testset = datasets.ImageFolder(root='Testing', transform=tf)
testset, valset = torch.utils.data.random_split(testset, [150, 244])

train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)
test_loader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=0)
val_loader = DataLoader(valset,batch_size=batch_size,shuffle=True,num_workers=0)

# model = models.resnet50() # we do not specify pretrained=True, i.e. do not load default weights
# model.fc = nn.Linear(2048, num_classes)

model = models.vgg16()
model.classifier[6] = nn.Linear(4096, num_classes)

model.load_state_dict(torch.load('VGG16.pt'))
# model.to(device)
model.eval()

y_pred = []
y_true = []

for images, labels in test_loader:

    images = images #.to(device)
    labels = labels #.to(device)

    outputs = model(images.float())
    _,prediction=torch.max(outputs.data,1)
    # print(prediction)
    # print('hoi')
    # print(labels)
    y_pred.extend(prediction)
    y_true.extend(labels)

#%%
y_pred_final = [int(x.item()) for x in y_pred]
y_true_final = [int(x.item()) for x in y_true]

#%%

classes = ('Glioma', 'Menigioma', 'No tumor', 'Pituitary')

cf_matrix = confusion_matrix(y_true_final, y_pred_final)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('cfmatrix.png')
# %%
print(trainset.class_to_idx)
# %%
