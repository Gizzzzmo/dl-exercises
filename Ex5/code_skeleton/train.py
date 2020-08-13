import torch as t
from data import ChallengeDataset, DataLoader
from trainer import Trainer, rotate_center
from matplotlib import pyplot as plt
import numpy as np
import model
import itertools
import random
import pandas as pd
from sklearn.model_selection import train_test_split

target_label = 0
stage2 = True

def count_parameters(model):
    return sum(p.numel() for p in model.new_params() if p.requires_grad)

to_three = lambda x: t.cat([x, x, x], dim=1)

batch_size = 50
# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
train_set, val_set = train_test_split(pd.read_csv('data.csv', sep=';'), random_state=0, train_size=0.75)
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_ch = ChallengeDataset(train_set, 'train')
val_ch = ChallengeDataset(val_set, 'val')
train_dl = t.utils.data.DataLoader(train_ch, batch_size=batch_size)
val_dl = t.utils.data.DataLoader(val_ch, batch_size=batch_size)

training_x = []
training_y = []
val_x = []
val_y = []

for i in range(len(train_ch)):
    x, y = train_ch[i]
    training_x.append(x[0])
    training_y.append(y[target_label:target_label+1])

for i in range(len(val_ch)):
    x, y = val_ch[i]
    val_x.append(x[0])
    val_y.append(y[target_label:target_label+1])

training_x = t.stack(training_x).reshape(-1, 1, 300, 300)
training_y = t.stack(training_y)
val_x = t.stack(val_x).reshape(-1, 1, 300, 300)
val_y = t.stack(val_y)

def feature_transform(x, prenet, batch_size):
    prenet.eval()
    features = []
    for i in range(0, len(x), batch_size):
        print(i)
        shape = x[i:i+batch_size].shape
        features.append(t.empty((shape[0], 16)))
        batch = x[i:i+batch_size]
        for j in range(4):
            batch = batch.transpose(2, 3)
            out = prenet(to_three(batch).cuda()).detach().cpu().clone()
            features[-1][:, 2*j:2*j+1] = out
            del out
            batch = batch.flip(dims=(2,))
            out = prenet(to_three(batch).cuda()).detach().cpu().clone()
            features[-1][:, 2*j+1:2*j+2] = out
    return t.cat(features, dim=0)

train_dl = DataLoader(training_x, training_y, batch_size, True, f=to_three, augment=True)
val_dl = DataLoader(val_x, val_y, batch_size, f=to_three)
# create an instance of our ResNet model
rn = model.SpecialModel()

print(rn)
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.BCEWithLogitsLoss()
optimizer = t.optim.Adadelta(rn.new_params(), lr=1e-0)
print(count_parameters(rn))
trainer = Trainer(rn, criterion, optimizer, train_dl, val_dl, cuda=True, early_stopping_patience=50, batches=10, augment=False)

# go, go, go... call fit on trainer
res = trainer.fit(200)
# plot the results

plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
