from torch import nn


class PaperEncoder(nn.Module):
    """
    Encoder architecture from "Representation Learning with CPC" paper
    """
    def __init__(self, vocab_size, embedding_dim=620, encoder_dim=2400, sentence_len=32, pad_idx=0, kernel_size=1):
        super(PaperEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.sentence_len = sentence_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv_block = nn.Sequiential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=encoder_dim, kernel_size=kernel_size),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=sentence_len)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # TODO: check that transpose is necessary
        feature_vector = self.conv_block(x)

        return feature_vector