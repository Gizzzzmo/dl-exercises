import torch as t
from trainer import Trainer
import sys
import torchvision as tv

epoch = int(sys.argv[1])
from model import OrigResNext

model = OrigResNext(True)
print(model)
crit = t.nn.BCEWithLogitsLoss()
trainer = Trainer(model, crit, cuda=False)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
