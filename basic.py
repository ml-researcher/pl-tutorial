# model and data -> fit!
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5,],std=[0.5,])])
data_train = datasets.MNIST(root = "./",
                            transform=transform,
                            train = True,
                            download = True)

from torch.utils.data import IterableDataset
from torch.distributed import get_rank
class IData(IterableDataset):
    def __init__(self):
        self.data = datasets.MNIST(root = "./",
                            transform=transform,
                            train = True,
                            download = True)
        self.cnt = 0
    def __iter__(self):
        for x in self.data:
            self.cnt += 1
            yield x
            if self.cnt >= 16:
                break

from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# model必须用lightningmodule，data可以用torch.utils.data.DataLoader也可以用pl.DataModule

# 之所以能看到训练的现象，是因为一些默认参数被省略了
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html

from torch.utils.data import DataLoader

train_dl = DataLoader(IData(), batch_size=2)#, shuffle=True)

from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=1, accelerator="gpu", devices=2, num_nodes=2, strategy="ddp", precision=16)
model = Model()
trainer.fit(model, train_dl)
# torchrun --nnodes 2 --nproc_per_node 2 --master_addr 192.255.250.15 --master_port 12345 --node_rank 0 basic.py