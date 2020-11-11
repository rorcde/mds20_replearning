from torch import nn
from .encoder import PaperEncoder


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

    def forward(self, past_sentences_enc):
        context = self.GRU(past_sentences_enc)
        return context


class CPC(nn.Module):
    def __init__(self, config):
        super(CPC, self).__init__()

        self.vocab_size = config['vocab_size']
        self.emb_dim = config.get('emb_dim', 620)
        self.enc_dim = config.get('enc_dim', 2400)
        self.ar_dim = config.get('ar_dim', self.enc_dim)
        self.max_sen_len = config['max_sen_len']

        self.encoder = PaperEncoder(
            self.vocab_size,
            self.emb_dim,
            self.enc_dim,
            self.max_sen_len,
            config['pad_idx'],
            config['kernel_size']
        )

        self.ar = AutoRegressiveModel(config)

        self.bilinear = nn.Bilinear(  # TODO: check that dimensions or OK. If not OK, replace by linear
            in1_features=self.enc_dim,
            in2_features=self.ar_dim,
            out_features=1,
            bias=False
        )

    def forward(self):
        pass
