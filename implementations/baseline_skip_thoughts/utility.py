import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import text_process_utility

def read_config_file(config_filename="config.json"):
    with open('config.json', 'r') as jsonfile:
        jsstring = jsonfile.read() 
        config_dict = json.loads(jsstring.replace('True', 'true'))
    return config_dict

config = read_config_file()
CUDA_DEVICE = config["cuda_device"]


def orthogonal_initialization(gru_cell, gain=1):
    # https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605/2
    # https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5
    # https://pytorch.org/docs/stable/nn.init.html

    """
    Orthogonal initialization of recurrent weights
    GRU cell contains 3 matrices in one parameter, so we slice it.
    gain - optional scaling factor (default=1)
    """

    with torch.no_grad():
        for _, hh, _, _ in gru_cell.all_weights:
            for i in range(0, hh.size(0), gru_cell.hidden_size):
                nn.init.orthogonal_(hh.data[i: i + gru_cell.hidden_size], gain=gain)
                

def reverse_variables(var):
    idx = Variable(torch.arange(var.size(0) - 1, -1, -1).long())
    idx = idx.cuda(1)
    inverted_var = var.index_select(0, idx)
    return inverted_var
                
def expand_tensor(tensor_to_expand):
    sizes = tensor_to_expand.size()
    expansion_size = (1, sizes[1], sizes[2])
    expanded_tensor = torch.cat([torch.zeros(expansion_size).cuda(CUDA_DEVICE), tensor_to_expand[:-1, :, :]])
    return expanded_tensor

def create_mask(var, lengths):
    mask = torch.zeros(var.size()).int()
    for i, l in enumerate(lengths):
        mask[i, :l] = 1
        
    mask = Variable(mask).float()
    mask = mask.cuda(CUDA_DEVICE)
    return mask


global losses_validation
global last_best_loss
losses_validation = []
last_best_loss = None

def check_model(i, loss, previous_sentence, next_sentence, predicted_previous_ids, predicted_next_ids):
    
    curr_loss = loss.item()
    losses_validation.append(curr_loss)
    losses_validation = losses_validation[-20:]
    print("Iteration {}: last_best_loss = {}, current_loss = {}".format(i, last_best_loss, curr_loss))
    
    print("Previous sentence = {}\n Next_sentence = {}\n Predicted previous sentence = {}\n Predicted next sentence = {}".format(
        text_process_utility.decode_sentences(previous_sentence, d.word_dictionary),
        text_process_utility.decode_sentences(next_sentence,  d.word_dictionary),
        text_process_utility.decode_sentences(predicted_previous_ids,  d.word_dictionary),
        text_process_utility.decode_sentences(predicted_next_ids,  d.word_dictionary)
    ))
    
    try:
        curr_mean_losses = sum(losses_validation)/len(losses_validation)
        if last_best_loss is None or last_best_loss > curr_mean_losses:
            print("Loss improved from {} to {}".format(last_best_loss, curr_mean_losses))
            
            save_loc = "./saved_models/skip-best"
            print("Saving model at {}".format(save_loc))
            torch.save(mod.state_dict(), save_loc)
            
            last_best_loss = curr_mean_losses
    except Exception as e:
       print("Couldn't save model because {}".format(e))
