from torch import nn


class PaperEncoder(nn.Module):
    """
    Encoder architecture from "Representation Learning with CPC" paper
    """
    def __init__(self, vocab_size, embedding_dim=620, weights=None, encoder_dim=2400, sentence_len=32, pad_idx=0,
                 kernel_size=1, **kwargs):
        super(PaperEncoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.sentence_len = sentence_len

        if weights is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(weights)  # place for your word2vec embeddings

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=encoder_dim, kernel_size=kernel_size),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=sentence_len)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # TODO: check that transpose is necessary
        feature_vector = self.conv_block(x)

        return feature_vector.squeeze(-1)


class RecurrentEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=620, weights=None, encoder_dim=2400, sentence_len=20, pad_idx=0,
                 **kwargs):
        super(RecurrentEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.sentence_len = sentence_len
        if weights is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(weights)  # place for your word2vec embeddings

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.encoder_dim,
            batch_first=True
        )

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.rnn(x)
        return h
