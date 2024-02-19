# Lightning Data Modul Kullanımı

Data modulü **prepare_data, setup, train_dataloader, val_dataloader, test_dataloader**  
fonksiyonlarını içerir.  
**prepare_data** fonksiyonu tekli gpular'da çalışır.
**setup** fonksiyonu çoklu gpular'da çağrılır.

```python
class MyDataModule(LightDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        self.mnist = MNIST(
            root = '/home/username/.cache/mnist',
            download = True
        )

    def setup(self):
        self.mnist = MNIST(
            root = '/home/username/.cache/mnist',
            download = True
        )
        
    def train_dataloader(self):
        return DataLoader(self.mnist.train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.mnist.train, batch_size=64)
    
    def test_dataloader(self):
        return DataLoader(self.mnist.train, batch_size=64)

```
