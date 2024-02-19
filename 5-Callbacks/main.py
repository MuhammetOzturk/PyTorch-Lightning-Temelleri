import torch
from lightning.pytorch.callbacks import Callback
from pytorch_lightning import LightningModule, LightningDataModule

class Counter(Callback):
    def __init__(self, epochs= 'epochs', verbose = True):
        super().__init__()
        self.epochs = epochs
        self.verbose = verbose
        self.state = {'epochs':0, 'batches': 0}

    @property
    def state_key(self) -> str : 
        return f"Counter[epochs = {self.epochs}]"

    def on_train_epoch_end(self, *args, **kwargs):
        if self.epochs == "epochs":
            self.state['epochs'] += 1

    def load_state_dict(self, state_dict):
        self.state.update = state_dict

    def state_dict(self):
        return self.state.copy()


