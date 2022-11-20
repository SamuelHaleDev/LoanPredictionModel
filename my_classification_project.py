
# IMPORTS


#IMPORTS
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

"""# DATA SET AND DATA LOADERS"""

df_raw = pd.read_csv(r'/content/Loan_Data.csv') #614x13 size

print(df_raw)

df_raw.drop(columns=['Loan_ID'], axis=1, inplace=True) #614x12 Size

print(df_raw)

num_rows = 614
#How many columns does it have? 12
num_cols = 12
#What are the column titles of the input variables?
input_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
              'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
#Which of the input columns or non numeric/ categorical
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                    'Property_Area', 'Loan_Status']
#What are the column names of the output/target data
output_cols = ['Loan_Status']

df_raw=df_raw.dropna(subset=['LoanAmount'])

df_raw=df_raw.dropna().reset_index(drop=True)

#CLEAN NANS FROM DATAFRAME
for catName in input_cols:
    df_raw[catName] = df_raw[catName].fillna(0)

print(df_raw) #After cleaning the NANS out we have size of 480x12

def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outputs as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array

#SEND IN RAW DATAFRAME AND GET NUMPY ARRAYS BACK
inputs_array, targets_array = dataframe_to_arrays(df_raw)

print(inputs_array)
print(targets_array)

inputs_array = inputs_array.astype(np.float32)
targets_array = targets_array.astype(np.int_)

print(inputs_array.dtype)
print(targets_array.dtype)

#NORMALIZE DATA AT THIS POINT FOR TRAINING
def NormalizeInput(input, cols):
    i = 0
    while i < len(cols):
        input[:, i] = input[:, i] / input[:, i].max()
        i = i + 1
    return input
inputs_array = NormalizeInput(inputs_array, input_cols)

print(inputs_array)

#CONVERT NP ARRAYS TO PYTORCH TENSORS
inputs = torch.from_numpy(inputs_array) #480x11 size
targets = torch.from_numpy(targets_array) #480x1 size

print(inputs.shape)
print(targets.shape)

#INSTANTIATE OUR DATALOADERS 400 TRAIN, 80 VAL
dataset = TensorDataset(inputs, targets)
train_ds, val_ds = random_split(dataset, [400, 80])
batch_size = 32

train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size*2)

"""# MODEL DEFINITION"""

class LoanEligibilityPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        targets = torch.flatten(targets)
        out = self(inputs)                  # Generate predictions
        loss = F.cross_entropy(out, targets) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        targets = torch.flatten(targets) 
        out = self(inputs)                    # Generate predictions
        loss = F.cross_entropy(out, targets)   # Calculate loss
        acc = accuracy(out, targets)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = LoanEligibilityPredictor(11, 2)

"""# TRAINING"""

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD, grad_clip=None, weight_decay=0):
    history = []
    optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()

            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

opt_func = torch.optim.SGD

grad_clip = 0

weight_decay = 0

evaluate(model, val_dl)

history = fit(200, .009, model, train_dl, val_dl, opt_func, grad_clip, weight_decay)

torch.save(model.state_dict(), 'my-classification-project.pth')

"""# Conclusions
- I will explain what I learned through discussing each challenge I faced in this project

1) My first challenge was figuring out how to get the dataset imported
-I couldn't figure out how to get a working download link working for the dataset
-I attempted to set one up through github.gist by pasting the csv's but I couldn't get them formatted correctly from excel
-Ended up settling on pandas read from csv and placing the path to the file locally

2) My second challenge was cleaning the data
- I had an extra column that wasn't input nor output it was simply identification for the loan which the model does not need so I dropped it.
- The data had Nan values in places it shouldn't have (i.e. people that got accepted for loans had a Nan loan value). Recognizing that this would negatively affect the model's ability to learn I decided to drop these rows
- A second miniature challenge I would relate to data cleansing is recognizing the output data and what it is representing. I started out thinking this was a regression problem and treated the data as so. As result I had the output array as float32 and this gave me a lot of errors and issues with training that I could not figure out. I had to delete my first attempt at this before I realized that was my biggest issue. 

3) My third challenges were what model architecture I would go for.
-I tried a lot of different implementations which I will list.
i) Feed forward with multiple hidden layers
ii) Tried sigmoid and relu activations
iii) Tried multiple loss functions like mse, cross entropy, binary cross entropy

4) My fourth challenge were the hyper parameters in which I tried a ton of.

I got negative results when I attempted to implement learning rate scheduling, weight decay and gradient clipping. So I did without those. I also attempted other optimizers but settled with SGD.

Overall I learned to be more thorough in planning an ML project. I learned about a lot of the different options I have when it comes to creating different architectures. 

The thing I did best was building the program incrementally and reusing code when I didn't have to create it from scratch. I made sure to run the program after every line of code I wrote which in turn led to more steady progress. It helped me take care of errors and problems as I ran into them instead of dealing with them all at the end.
"""