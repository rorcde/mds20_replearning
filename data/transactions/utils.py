from nltk import wordpunct_tokenize
from torch.utils import data
import pandas as pd
from ..language.utils import build_vocabulary, default_collate_fn


class TransactionsDataset(data.Dataset):
    def __init__(self, transactions: pd.DataFrame, mcc_codes: pd.DataFrame, gender: pd.DataFrame,
                 w2i_mapping=None, vocab_size=None, pad_token='[PAD]',
                 unk_token='[UNK]', eos_token='[EOS]',
                 mcc_code_col_name='mcc_code', client_id_col_name='customer_id', gender_col_name='gender',
                 mcc_descriptions_col_name='mcc_description',
                 sep=' '):
        super(TransactionsDataset, self).__init__()

        self.mcc_code_col_name = mcc_code_col_name
        self.mcc_descriptions_col_name = mcc_descriptions_col_name
        self.client_id_col_name = client_id_col_name
        self.gender_col_name = gender_col_name
        self.sep = sep

        data_ = transactions.join(mcc_codes.set_index(mcc_code_col_name), on=mcc_code_col_name)
        data_ = data_.join(gender.set_index(client_id_col_name), on=client_id_col_name)

        self.data_ = data_.groupby(client_id_col_name)[mcc_descriptions_col_name, gender_col_name].agg(
            {mcc_descriptions_col_name: lambda x: ' '.join(x), gender_col_name: lambda x: x.iloc[0]}).dropna()

        if w2i_mapping is not None:
            self.w2i_mapping = w2i_mapping
            self.vocab_size = len(w2i_mapping)
        else:
            self.w2i_mapping = build_vocabulary(self.data_[mcc_descriptions_col_name].values, vocab_size,
                                                pad_token, unk_token, eos_token)
            self.vocab_size = vocab_size

        self.pad_idx = self.w2i_mapping[pad_token]
        self.unk_idx = self.w2i_mapping[unk_token]
        self.eos_idx = self.w2i_mapping[eos_token]

    def __len__(self):
        return self.data_.shape[0]

    def __getitem__(self, item):
        sample = self.data_.iloc[item]
        return [self.w2i_mapping.get(w, self.unk_idx) for w in wordpunct_tokenize(sample[self.mcc_descriptions_col_name])], \
            sample[self.gender_col_name]
