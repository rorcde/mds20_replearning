from torch import nn
from torch.optim import Adam

import pytorch_lightning as pl

from copy import deepcopy


class ClassificationArchitecture(nn.Module):
    def __init__(self, encoder_instance, n_classes=1, freeze=False):
        super(ClassificationArchitecture, self).__init__()
        self.encoder_instance = deepcopy(encoder_instance)
        self.encoder_dim = self.encoder_instance.encoder_dim
        self.n_classes = n_classes

        self.freeze = freeze
        if self.freeze:
            for param in self.encoder_instance.parameters():
                param.requires_grad = False

        self.linear_head = nn.Linear(self.encoder_dim, self.n_classes)

    def forward(self, x):
        x = self.encoder_instance(x)
        x = self.linear_head(x)
        if self.n_classes == 1:
            return x.squeeze(-1)
        return x


class ClassificationModule(pl.LightningModule):
    def __init__(self, pretrained_encoder, freeze,  lr):
        super(ClassificationModule, self).__init__()
        self.lr = lr
        self.model = ClassificationArchitecture(encoder_instance=pretrained_encoder, n_classes=1, freeze=freeze)
        self.valid_acc = pl.metrics.classification.Accuracy(threshold=0)  # zero threshold due to logits format of the output

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        text_batch = batch[0]
        label_batch = batch[1]

        logits = self.model(text_batch)
        cross_entropy = nn.functional.binary_cross_entropy_with_logits(logits, label_batch.float())

        self.log('cross_entropy', cross_entropy, prog_bar=True, logger=True)
        return cross_entropy

    def validation_step(self, batch, batch_idx):
        text_batch = batch[0]
        label_batch = batch[1]

        logits = self.model(text_batch)
        cross_entropy = nn.functional.binary_cross_entropy_with_logits(logits, label_batch.float())

        self.log('val_cross_entropy', cross_entropy, prog_bar=True, logger=True)
        self.log('val_accuracy', self.valid_acc(logits, label_batch), prog_bar=True, logger=True)

        return cross_entropy





