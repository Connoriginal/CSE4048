import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import BaseModel, BILSTM, LSTM_ATTENTION
from dataset import TextDataset, make_data_loader
from sklearn.metrics import classification_report
from vocab import Vocabulary


def test(args, data_loader, model):
    true = np.array([])
    pred = np.array([])
    model.eval()
    for i, (text, label) in enumerate(tqdm(data_loader)):
        input_lengths = torch.tensor([len(x.nonzero()) for x in text])
        input_lengths, perm_idx = input_lengths.sort(0,descending=True)
        text = text[perm_idx]
        label = label[perm_idx]

        text = text.to(args.device)
        label = label.to(args.device)            

        output, _ = model(text,input_lengths)
        
        label = label.squeeze()
        output = output.argmax(dim=-1)
        output = output.detach().cpu().numpy()
        pred = np.append(pred,output, axis=0)
        
        label = label.detach().cpu().numpy()
        true =  np.append(true,label, axis=0)

    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument('--vocab_size', type=int, default=60000, help="maximum vocab size")
    parser.add_argument('--model_name', type=str, default='model.pt')
    parser.add_argument('--batch_first', action='store_true', default=True,help="If true, then the model returns the batch first")

    args = parser.parse_args()

    """
    TODO: You MUST write the same model parameters as in the train.py file !!
    """
    # Model parameters
    input_size = args.vocab_size
    output_size = 4     # num of classes
    embedding_dim = 300 # embedding dimension
    hidden_dim = 64  # hidden size of RNN
    num_layers = 3
        

    # Make Test Loader
    test_dataset = TextDataset(args.data_dir, 'test', args.vocab_size)
    args.pad_idx = test_dataset.sentences_vocab.wtoi['<PAD>']
    test_loader = make_data_loader(test_dataset, args.batch_size, args.batch_first, shuffle=False)

    sentences_vocab = Vocabulary(args.vocab_size)
    sentences_vocab.load_vocabulary('data','./pickles')
    args.pad_idx = sentences_vocab.wtoi['<PAD>']
    args.unk_idx = sentences_vocab.wtoi['<UNK>']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # instantiate model
    # model = BaseModel(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    # model = BILSTM(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    model = LSTM_ATTENTION(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first, device= args.device)
    model.load_state_dict(torch.load(args.model_name, map_location=device))
    model = model.to(device)
    
    print(test_dataset.labels_vocab.itow)
    target_names = [ w for i, w in test_dataset.labels_vocab.itow.items()]
    # Test The Model
    pred, true = test(args, test_loader, model)
    
    
    accuracy = (true == pred).sum() / len(pred)
    print("Test Accuracy : {:.5f}".format(accuracy))

    ## Save result
    strFormat = '%12s%12s\n'

    with open('result.txt', 'w') as f:
        f.write('Test Accuracy : {:.5f}\n'.format(accuracy))
        f.write('true label  |  predict label \n')
        f.write('-------------------------- \n')
        
        for i in range(len(pred)):
            f.write(strFormat % (test_dataset.labels_vocab.itow[true[i]],test_dataset.labels_vocab.itow[pred[i]]))
            
  
    print(classification_report(true, pred, target_names=target_names))
    