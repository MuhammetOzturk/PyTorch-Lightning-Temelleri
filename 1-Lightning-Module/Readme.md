# Pytorch Lightning Modül Kullanımı
Lightning modülünün temel fonksiyonları **training_step,validate_step, test_step
configure_optimizers** fonksiyonlarıdır.  
Kendi modelimizi oluştururken bu fonksiyonlar bulunmalıdır.

```python
class MnistModel(LightningModule):
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

    def training_step(self,batch, batch_idx):
       loss, predicts = self._common_step(batch, batch_idx)
       return loss

    def validate_step(self,batch, batch_idx):
       loss, predicts = self._common_step(batch, batch_idx)
       return loss

    def test_step(self,batch, batch_idx):
       loss, predicts = self._common_step(batch, batch_idx)
       return loss

    def _common_step(self, batch , batch_idx):
        x,y = batch
        x = x.view(x.size(0), -1)
        predicts = self(x)
        loss = nn.functionaln.cross_entropy(x, y)
        return loss, predicts
```



