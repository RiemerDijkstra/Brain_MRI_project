# train vgg16 on fmri data

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

from sklearn.metrics import accuracy_score,classification_report

from tqdm import tqdm_notebook as tqdm
import time
import warnings
warnings.simplefilter("ignore")

IMAGE_SIZE=(128,128)
batch_size= 20
learning_rate = 1.1682013821438381e-05
epochs=20
num_classes=4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

print(torch.cuda.get_device_name(0))

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

print(trainset.class_to_idx)

visualize_loader = DataLoader(testset,batch_size=6,shuffle=True,num_workers=0)

batch = next(iter(visualize_loader))
images, labels  = batch
grid = utils.make_grid(images, nrow = 3)
plt.figure(figsize=(11,11))
plt.imshow(np.transpose(grid, (1,2,0)))
print('labels: ', labels)

model = models.vgg16(pretrained=True)

model.classifier[6] = nn.Linear(4096, num_classes)

model=model.to(device)
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(),lr=learning_rate)

#%%
# NEW TRAIN FUNCTION TO SEE IF ITS CORRECT
best_accuracy=0.0

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

since = time.time()

for epoch in range(epochs):
    
    #Evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    val_loss = 0.0
    
    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=images.to(device)
            labels=labels.to(device)
            
        optimizer.zero_grad()
        
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
        train_loss+= loss.item()
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        
    train_accuracy=train_accuracy/len(trainset)
    train_loss=train_loss/len(trainset)
    
    train_acc_list.append(train_accuracy)
    train_loss_list.append(train_loss)
    
    # Evaluation on testing dataset
    with torch.no_grad():
        model.eval()
        
        val_accuracy=0.0
        for i, (images,labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                images=images.to(device)
                labels=labels.to(device)
                
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            outputs=model(images)
            _,prediction=torch.max(outputs.data,1)
            val_accuracy+=int(torch.sum(prediction==labels.data))
            val_loss += loss.item()
    
        val_accuracy=val_accuracy/len(valset)
        val_loss = val_loss/len(valset)

    val_acc_list.append(val_accuracy)
    val_loss_list.append(val_loss)
    
    
    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+ ' Validation Loss: '+str(val_loss)+' Validation Accuracy: '+str(val_accuracy))
    
    if val_accuracy>best_accuracy:
        torch.save(model.state_dict(), 'other_training2.pt')
        print('MODEL SAVED')
        best_accuracy=val_accuracy
time_elapsed = time.time() - since
print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best validation accuracy: {:4f}'.format(best_accuracy))

#%%

# DO HYPERPARAMETER TUNING WITH OWN TRAIN AND EVAL FUNCTION

def train_and_evaluate(param, model):
    EPOCHS = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])
    criterion=nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.cuda()

    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    trainset = datasets.ImageFolder(root='Training', transform=tf)
    testset = datasets.ImageFolder(root='Testing', transform=tf)
    testset, valset = torch.utils.data.random_split(testset, [150, 244])

    val_data_length = len(valset)

    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)
    test_loader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader = DataLoader(valset,batch_size=batch_size,shuffle=True,num_workers=0)

    train_dataloader = train_loader
    val_dataloader = val_loader

    for epoch_num in range(EPOCHS):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in train_dataloader:

                train_label = train_label.to(device)
                train_input = train_input.to(device)

                output = model(train_input.float())
                    
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                    
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    val_input = val_input.to(device)

                    output = model(val_input.float())

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                        
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                
            accuracy = total_acc_val/val_data_length

    return accuracy


def objective(trial):
    params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
            }
        
    model = models.vgg16(pretrained=True)

    model.classifier[6] = nn.Linear(4096, num_classes)
        
    accuracy = train_and_evaluate(params, model)

    return accuracy

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=10)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))

#%%

# TEST AND PLOT RESULTS

def test(model,testloader):
    with torch.no_grad():
        n_correct=0
        n_samples=0
        y_pred=[]
        y_actual=[]
        for i,(images,labels) in enumerate(testloader):
            images=images.to(device)
            labels=labels.to(device)
                
            outputs=model(images)
                
            y_actual+=list(np.array(labels.detach().to('cpu')).flatten())
        # value ,index
            _,predictes=torch.max(outputs,1)
            y_pred+=list(np.array(predictes.detach().to('cpu')).flatten())
        # number of samples in current batch
            n_samples+=labels.shape[0]

            n_correct+= (predictes==labels).sum().item()
                
        y_actual=np.array(y_actual).flatten()
        y_pred=np.array(y_pred).flatten()
        print(np.unique(y_pred))
        acc = classification_report(y_actual,y_pred,target_names=trainset.classes)
        print(f"{acc}")



test(model,test_loader)
plt.figure(figsize=(20,5))
plt.plot(train_loss_list, '-o', label="train")
plt.plot(val_loss_list, '-o', label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss change over epoch")
plt.legend()

plt.figure(figsize=(20,5))
plt.plot(train_acc_list, '-o', label="train")
plt.plot(val_acc_list, '-o', label="val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over epoch")
plt.legend()
# %%
