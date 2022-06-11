import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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