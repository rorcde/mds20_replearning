import torch
import utility
from collections import OrderedDict

config = utility.read_config_file()
USE_CUDA = config["use_cuda"]
CUDA_DEVICE = config["cuda_device"]


def encode_sentence(sentence, word_dictionary, vocabulary_dim, max_len, idx_EOS=0, idx_UNK=1):
    indices = []
    for w in sentence.split():
        if word_dictionary.get(w, vocabulary_dim + 1) < vocabulary_dim:
            idx = word_dictionary.get(w)
        else:
            idx = idx_UNK
        indices.append(idx)
    indices = indices[: max_len - 1]
    indices += [idx_EOS] * (max_len - len(indices))

    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda(CUDA_DEVICE)
    return indices


def decode_index(idx, word_dictionary, idx_EOS=0, idx_UNK=1):
    idx = idx.item()
    if idx == idx_EOS:
        return "EOS"
    elif idx == idx_UNK:
        return "UNK"

    search_idx = idx - 2
    if search_idx >= len(word_dictionary):
        return "NA"

    word = list(word_dictionary.keys())[list(word_dictionary.values()).index(search_idx)]

    return word


def decode_sentences(indices, reverse_word_dictionary):
    words = [decode_index(idx, reverse_word_dictionary) for idx in indices]
    return " ".join(words)


def build_dictionary(sentences):
    wordcount = {}
    for s in sentences:
        words = s.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1

    sorted_words = sorted(list(wordcount.keys()), key=lambda x: wordcount[x], reverse=True)

    worddict = OrderedDict()
    for idx, word in enumerate(sorted_words):
        worddict[word] = idx + 2
    worddict["EOS"] = 0
    worddict["UNK"] = 1

    return worddict


#def build_dictionary(sentences, pad_token='PAD',
#                     unk_token='UNK', eos_token='EOS'):
#    list_of_words = []
#    for sentence in sentences:
#        for word in sentence.split():
#            list_of_words.append(word)

#    all_words = [pad_token, unk_token, eos_token] + list(set(list_of_words))
#    word_dictionary = {w: i for i, w in enumerate(all_words)}
#    return word_dictionary




