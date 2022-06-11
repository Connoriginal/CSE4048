import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class BaseModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first):
        super(BaseModel, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim

        """
        TODO: Implement your own model. You can change the model architecture.
        """
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim, output_size)

    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x, input_lengths):
        batch_size = x.size(0) if self.batch_first else x.size(1)

        #h_0: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        embedding = self.embedding(x)
        packed_input = pack_padded_sequence(embedding, input_lengths.tolist(), batch_first=self.batch_first)
        # output, hidden = self.rnn(packed_input, (h_0, c_0))
        packed_output, hidden = self.rnn(packed_input)
        # output, _ = pad_packed_sequence(output, batch_first=self.batch_first)

        output = self.fc(hidden[0][-1])

        # outputs, hidden = self.rnn(embedding, None)  # outputs.shape -> (sequence length, batch size, hidden size)

        # outputs = outputs[:, -1, :] if self.batch_first else outputs[-1, :, :]
        # output = self.fc(outputs)
        
        return output, hidden

class BILSTM(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first):
        super(BILSTM, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim

        """
        TODO: Implement your own model. You can change the model architecture.
        """
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim*2, output_size)

    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x, input_lengths):
        batch_size = x.size(0) if self.batch_first else x.size(1)

        #h_0: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        embedding = self.embedding(x)
        packed_input = pack_padded_sequence(embedding, input_lengths.tolist(), batch_first=self.batch_first)
        # output, hidden = self.rnn(packed_input, (h_0, c_0))
        packed_output, hidden = self.rnn(packed_input)
        h = torch.cat((hidden[0][-2], hidden[0][-1]), dim=1)
        # output, _ = pad_packed_sequence(output, batch_first=self.batch_first)

        output = self.fc(h)
        
        return output, hidden

class LSTM_ATTENTION(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first, device = 'cpu'):
        super(LSTM_ATTENTION, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim
        self.device = device

        """
        TODO: Implement your own model. You can change the model architecture.
        """
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=batch_first)
        self.W = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_dim*2, output_size)

    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x, input_lengths):
        batch_size = x.size(0) if self.batch_first else x.size(1)

        #h_0: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_dim)
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)

        embedding = self.embedding(x)
        packed_input = pack_padded_sequence(embedding, input_lengths.tolist(), batch_first=self.batch_first)
        packed_output, hidden = self.rnn(packed_input, (h_0, c_0))
        output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)

        # initial input <SOS> : 2 for decoder
        inp = torch.zeros(output.size(0), 1).long()
        inp += 2
        inp = inp.to(self.device)
        emb = self.embedding(inp)
        dec_out, dec_hid = self.rnn(emb, hidden)

        # attention
        dec_out_view = dec_out.view(dec_out.shape[0], dec_out.shape[2],-1)
        at_score = torch.bmm(output, dec_out_view)
        at_dis = F.softmax(at_score, dim=1)
        at_val = torch.sum(at_dis * output, dim=1)

        con = torch.cat((at_val, dec_out.squeeze(1)), dim=1)
        output = self.fc(self.tanh(self.W(con)))

        
        return output, hidden