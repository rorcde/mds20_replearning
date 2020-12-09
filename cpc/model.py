import torch
from torch import nn
from .encoder import PaperEncoder
from .nce import info_nce


class AutoRegressiveModel(nn.Module):
    def __init__(self, enc_dim=2400, ar_dim=2400):
        super(AutoRegressiveModel, self).__init__()

        self.enc_dim = enc_dim
        self.ar_dim = ar_dim

        self.GRU = nn.GRU(
            input_size=self.enc_dim,
            hidden_size=self.ar_dim,
            batch_first=True
        )

        self.Wk = nn.Linear(self.ar_dim, self.enc_dim)

    def forward(self, past_sentences_enc):
        context, _ = self.GRU(past_sentences_enc)
        return self.Wk(context)


class CPC(nn.Module):
    def __init__(self, vocab_size, emb_dim=620, enc_dim=2400, ar_dim=2400, kernel_size=1, pad_idx=0,
                 reshape_before_ar=1):
        super(CPC, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_dim = enc_dim
        self.ar_dim = ar_dim
        self.reshape_before_ar = reshape_before_ar

        self.encoder = PaperEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=self.emb_dim,
            encoder_dim=self.enc_dim,
            pad_idx=pad_idx,
            kernel_size=kernel_size
        )

        self.ar = AutoRegressiveModel(self.enc_dim, self.ar_dim)

    def forward(self, batch, k=1):
        """
        batch - sequences x encoder_dim
        """
        enc = self.encoder(batch)

        enc = torch.reshape(enc, (self.reshape_before_ar, -1, self.enc_dim))  # reshape into batches to speed-up ar
        inp_enc, outp_enc = enc[:, :-k], enc[:, k:]

        predicted_enc = self.ar(inp_enc)

        return info_nce(predicted_enc, outp_enc)
