from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch.optim import Adam

from skip_thought_vectors.model import SkipThoughtModel


class SkipThoughtsModule(pl.LightningModule):
    def __init__(self, lr=3e-4, i2w_mapping=None):
        super(SkipThoughtsModule, self).__init__()
        self.save_hyperparameters()
        self.skipthoughts = SkipThoughtModel()
        self.skipthoughts.initialize_parameters()
        self.vocab_dim = self.skipthoughts.config['vocabulary_dim']
        self.i2w_mapping = None

        self.lr = lr

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def forward(self, batch):
        return self.skipthoughts(batch)

    def training_step(self, batch, batch_idx):

        output = self.skipthoughts(batch)
        loss = output[0]

        self.log('loss', loss, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        output = self.skipthoughts(batch)
        loss = output[0]

        # do metrics here using output
        self.log('val_loss', loss, prog_bar=True, logger=True)

        if batch_idx == 0:
            return output[3:]

    def validation_epoch_end(self, outputs):
        if self.i2w_mapping is not None:
            for name, output in zip(
                    ['previous_true', 'next_true', 'previous_pred', 'next_pred'],
                    outputs
            ):
                self.logger.experiment.add_text(
                    name,
                    ' '.join([self.i2w_mapping[w] for w in output])
                )
