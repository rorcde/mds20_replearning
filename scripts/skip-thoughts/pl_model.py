from collections import OrderedDict

import pytorch_lightning as pl
from torch.optim import Adam

from skip_thought_vectors.model import SkipThoughtModel


class SkipThoughtsModule(pl.LightningModule):
    def __init__(self, lr=3e-4):
        super(SkipThoughtsModule, self).__init__()
        self.save_hyperparameters()
        self.skipthoughts = SkipThoughtModel()
        self.skipthoughts.initialize_parameters()
        self.vocab_dim = self.skipthoughts.vocabulary_dim

        self.lr = lr

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def forward(self, batch):
        return self.skipthoughts(batch)

    def training_step(self, batch, batch_idx):

        output = self.skipthoughts(batch)
        loss = output[0]

        return OrderedDict({
            'loss': loss,
        })

    def validation_step(self, batch, batch_idx):

        output = self.skipthoughts(batch)
        loss = output[0]

        # do metrics here using output

        return OrderedDict({
            'loss': loss,
        })