from torch import nn
from .encoder import PaperEncoder
from .nce import info_nce


class AutoRegressiveModel(nn.Module):
    def __init__(self, config):
        super(AutoRegressiveModel, self).__init__()

        self.enc_dim = config.get('enc_dim', 2400)
        self.ar_dim = config.get('ar_dim', self.enc_dim)

        self.GRU = nn.GRU(
            input_size=self.enc_dim,
            hidden_size=self.ar_dim,
            batch_first=True
        )

        self.Wk = nn.Linear(self.ar_dim, self.self.enc_dim)

    def forward(self, past_sentences_enc):
        context, _ = self.GRU(past_sentences_enc)
        return self.Wk(context)


class CPC(nn.Module):
    def __init__(self, config):
        super(CPC, self).__init__()

        self.vocab_size = config['vocab_size']
        self.emb_dim = config.get('emb_dim', 620)
        self.enc_dim = config.get('enc_dim', 2400)
        self.ar_dim = config.get('ar_dim', self.enc_dim)
        self.max_sen_len = config['max_sen_len']

        self.encoder = PaperEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=self.emb_dim,
            encoder_dim=self.enc_dim,
            sentence_len=self.max_sen_len,
            pad_idx=config['pad_idx'],
            kernel_size=config['kernel_size']
        )

        self.ar = AutoRegressiveModel(config)

    def forward(self, batch, k=1):
        """
        batch - sequences x encoder_dim
        """
        enc = self.encoder(batch)
        inp_enc, outp_enc = enc[:-k], enc[k:]

        predicted_enc = self.ar(inp_enc.unsqueeze(0)).squeeze(0)  # TODO: batch learning

        return info_nce(predicted_enc, outp_enc)
