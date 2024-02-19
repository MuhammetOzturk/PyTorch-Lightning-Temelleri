import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        self.mnist_train = MNIST(
            root = '/home/muhammet/.cache/mnist/',
            download = True,
            train = True,
            transform = ToTensor()
        )

        self.mnist_test = MNIST(
            root = '/home/muhammet/.cache/mnist',
            download = True,
            train = False,
            transform = ToTensor()
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size = 64)

    def val_dataloader(self):
        return DataLoader(self.mnist_train, batch_size = 64)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size = 64)

class MyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256,10),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self,batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log_dict({'loss':loss}, on_epoch=True, on_step = False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log_dict({'loss':loss}, on_epoch=True, on_step = False)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log_dict({'loss':loss}, on_epoch=True, on_step = False)

    def _common_step(self, batch, batch_idx):
        x,y = batch 
        x = x.view(x.size(0), -1)
        x = self(x)
        loss = F.cross_entropy(x, y)
        return loss 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= 1e-3)


datamodule = MyDataModule()
model = MyModule()

trainer = pl.Trainer(max_epochs = 3)

trainer.fit(model, datamodule)
