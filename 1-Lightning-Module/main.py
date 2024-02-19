import torch 
from torch import nn
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as L

class MnistModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256,10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss , predicts = self._common_step(batch, batch_idx)
        self.log_dict({'loss':loss},on_step=False, on_epoch=True)
        return loss

    def validate_step(self, batch, batch_idx):
        loss, predicts = self._common_step(batch, batch_idx)
        self.log_dict({'loss':loss},on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss , _ = self._common_step(batch,batch_idx)
        self.log_dict({'loss':loss})

    def _common_step(self, batch, batch_idx):
        x,y = batch
        x = x.view(x.size(0), -1)
        predicts = self(x)
        loss = F.cross_entropy(predicts, y)
        return loss, predicts

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-3)

mnist_train = MNIST(
    root = '/home/muhammet/.cache/mnist/',
    train = True,
    transform = ToTensor()
)

mnist_test = MNIST(
    root = '/home/muhammet/.cache/mnist/',
    train = False,
    transform = ToTensor()
)

train,validate = random_split(mnist_train,[0.8,0.2])

train_data_loader = DataLoader(train, batch_size = 64)
validate_data_loader = DataLoader(validate, batch_size = 64)
test_data_loader = DataLoader(mnist_test, batch_size = 64)

my_model = MnistModule()

trainer = L.Trainer(max_epochs = 3)

trainer.fit(my_model, train_data_loader, validate_data_loader)
trainer.test(my_model, test_data_loader)
