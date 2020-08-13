import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm#
import hashlib
import numpy as np
import matplotlib.pyplot as plt


def rotate_center(x, dim0, dim1, deg):
    outshape = tuple(int(np.ceil(x.shape[k]*np.sqrt(2))) if k == dim0 or k == dim1 else x.shape[k] for k in range(len(x.shape)))

    deg *= -np.pi/180
    width = outshape[dim0]
    height = outshape[dim1]
    inverse = t.Tensor(
        [[np.cos(deg), -np.sin(deg)],
        [np.sin(deg), np.cos(deg)]]
    ).to(x.device)
    indices = t.arange(width)
    indices = t.stack([indices]*width)
    indices = t.stack([indices, indices.transpose(0, 1)], dim=1).to(x.device)
    rotated = (inverse @ (indices - (width-1)/2)) + (width-1)*(0.5 - 0.125*np.sqrt(2))

    i = rotated[:, 0, :].round().long()
    j = rotated[:, 1, :].round().long()
    sl1 = tuple(0 if dim0 == k or dim1 == k else slice(None) for k in range(len(x.shape)))
    save = x[sl1].clone()
    x[sl1] = 0

    outofbounds = (i < 0) | (j < 0) | (j >= x.shape[dim0]) | (i >= x.shape[dim1])
    i[outofbounds] = 0
    j[outofbounds] = 0

    sl2 = tuple(i if dim0 == k else (j if dim1 == k else slice(None)) for k in range(len(x.shape)))

    output = x[sl2]
    x[sl1] = save
    return output



class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1,
                 batches=1,
                 augment=False):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._epoch = 0
        self.batches = batches
        self.augment = augment

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/'+hashlib.md5(str(self._model).encode()).hexdigest()+'_best_checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        self._epoch = epoch_n
        ckp = t.load('checkpoints/'+hashlib.md5(str(self._model).encode()).hexdigest()+'_best_checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        self._optim.zero_grad()
        batch_loss = 0
        for i in range(self.batches):
            pred = self._model(x[i*len(x)//self.batches:(i+1)*len(x)//self.batches])
            # print(t.transpose(pred, 0, 1))
            # print(t.transpose(y, 0, 1))
            
            loss = self._crit(pred, y[i*len(x)//self.batches:(i+1)*len(x)//self.batches])
            # print(loss)
            loss.backward()
            batch_loss += loss.item()
        
        self._optim.step()
        
        return batch_loss
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        preds = []
        loss = 0
        for i in range(self.batches):
            output = self._model(x[i*len(x)//self.batches:(i+1)*len(x)//self.batches])
            # print(t.transpose(pred, 0, 1))
            # print(t.transpose(y, 0, 1))
            
            loss += self._crit(output, y[i*len(x)//self.batches:(i+1)*len(x)//self.batches]).item()
            # print(loss)
            preds.append(output > 0)
        
        return loss, t.cat(preds)

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        self._model.train()

        loss = 0
        samples = 0

        for i, (x, y) in enumerate(tqdm(self._train_dl, ncols=100, ascii=True)):
            if self._cuda:
                y = y.cuda()
            if self.augment:
                for i in range(0, 8):
                    mini_x = rotate_center(x.cuda() if self._cuda else x, 2, 3, i*360/8)
                    #plt.imshow(mini_x[0, 0].detach().cpu())
                    #plt.show()
                    loss += self.train_step(mini_x, y)
                    loss += self.train_step(mini_x.flip(dims=(2,)), y)
                    samples += 2*len(mini_x)
            else:
                loss += self.train_step(x.cuda() if self._cuda else x, y)
                samples += len(x)
            
        return loss/samples
    
    def plot_imgs(self, imgs, labels, predictions):
        rows = (len(imgs)+2)//3

        _, axs = plt.subplots(rows, 3)
        for i, (img, label, pred) in enumerate(zip(imgs, labels, predictions)):
            axs[i//3, i%3].axis('off')
            axs[i//3, i%3].imshow(img)
            axs[i//3, i%3].set_title('prediction: ' + str(label.item()) + ', gt: ' + str(pred.item()))

    def val_test(self, plot_failures=False, dl=None):
        if dl is None:
            dl = self._val_test_dl
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        self._model.eval()

        correct_cracks = 0
        correct_non_cracks = 0

        predicted_cracks = 0

        cracks = 0

        samples = 0

        

        with t.no_grad():
            loss = 0

            crack_failure_images = []
            failed_crack_predictions = []
            actual_crack_labels = []

            for x, y in tqdm(dl, ncols=100, ascii=True):
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                if self.augment:
                    x = rotate_center(x, 2, 3, 0)
                batch_loss, predictions = self.val_test_step(x, y)
                loss += batch_loss

                if plot_failures:
                    failures = (predictions != y)
                    crack_indices = failures[:, 0].nonzero()

                    crack_failure_images.append(x[crack_indices[:, 0]].cpu().detach())

                    failed_crack_predictions.append(predictions[crack_indices[:, 0], 0].cpu().detach())

                    actual_crack_labels.append(y[crack_indices[:, 0], 0].cpu().detach())
                    

                correct_cracks += ((predictions[:, 0] == 1) * (y[:, 0] == 1)).sum().float()

                correct_non_cracks += ((predictions[:, 0] == 0) * (y[:, 0] == 0)).sum().float()

                cracks += y[:, 0].sum().float()

                predicted_cracks += predictions[:, 0].sum().float()

                samples += len(predictions)
                
        crack_recall = correct_cracks/cracks

        crack_precision = correct_cracks/predicted_cracks

        crack_f1 = 2 * (crack_precision * crack_recall)/(crack_precision + crack_recall)

        crack_accuracy = (correct_cracks + correct_non_cracks)/samples
        print(correct_cracks.item(), predicted_cracks.item(), cracks.item(), samples)
        print('metrics:')
        print('recall:', crack_recall.item(), 'precision:', crack_precision.item(), 'f1:', crack_f1.item(), 'accuracy:', crack_accuracy.item())

        if plot_failures:
            failed_crack_predictions = t.cat(failed_crack_predictions)

            actual_crack_labels = t.cat(actual_crack_labels)

            crack_failure_images = t.transpose(t.cat(crack_failure_images), 1, 3)

            crack_failure_images -= crack_failure_images.min()

            crack_failure_images /= crack_failure_images.max()

            for i in range(0, len(crack_failure_images), 6):
                self.plot_imgs(crack_failure_images[i:i+6],  failed_crack_predictions[i:i+6], actual_crack_labels[i:i+6])
                plt.show()

        return loss/samples
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        
        train_losses = []
        validation_losses = []

        epochs += self._epoch
        smallest_loss = float('inf')
        stopping_counter = 0
        lr_counter = 0
        smallest_train_loss = float('inf')
        while self._epoch != epochs:
            train_losses.append(self.train_epoch())
            print('\n-------------------------------\n')
            validation_losses.append(self.val_test())
            print('')
            print(self._epoch, train_losses[-1], validation_losses[-1])
            print('\n-------------------------------\n')
            print(self.val_test(dl=self._train_dl))
            print('\n-------------------------------\n')
            if validation_losses[-1] < smallest_loss:
                self.save_checkpoint(self._epoch)
                smallest_loss = validation_losses[-1]
                stopping_counter = 0
            else:
                stopping_counter += 1
                if stopping_counter == self._early_stopping_patience:
                    break

            if train_losses[-1] < smallest_train_loss:
                lr_counter = 0
                smallest_train_loss = train_losses[-1]
            else:
                lr_counter += 1
                if lr_counter == 5:
                    print('dropping learn rate')
                    for param_group in self._optim.param_groups:
                        param_group['lr'] *= 0.1
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            self._epoch += 1
        return train_losses, validation_losses
                    
        
        
        