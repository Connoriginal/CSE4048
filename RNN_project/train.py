import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import BaseModel, BILSTM, LSTM_ATTENTION
from dataset import TextDataset, make_data_loader
from util import *
from vocab import Vocabulary
from torchtext.vocab import GloVe

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, valid_loader,model,emb_init_weight):
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    tensorboard_plt = TensorboardPlotter(args.log_dir)
    

    model.embedding.weight.data.copy_(emb_init_weight)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    min_loss = np.Inf
    
    for epoch in range(args.num_epochs):
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.num_epochs}]")
        
        model.train()
        for i, (text, label) in enumerate(tqdm(data_loader)):
            input_lengths = torch.tensor([len(x.nonzero()) for x in text])
            input_lengths, perm_idx = input_lengths.sort(0,descending=True)
            text = text[perm_idx]
            label = label[perm_idx]


            text = text.to(args.device)
            label = label.to(args.device)            
            optimizer.zero_grad()

            output, _ = model(text, input_lengths)
            
            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total
        

        # Plot to tensorboard
        tensorboard_plt.loss_plot('loss','train',epoch_train_loss,epoch)
        tensorboard_plt.loss_plot('accuracy','train',epoch_train_acc,epoch)


        # Validation
        valid_losses = []
        valid_acc = 0.0
        total = 0
        model.eval()
        for i, (text, label) in enumerate(tqdm(valid_loader)):
            input_lengths = torch.tensor([len(x.nonzero()) for x in text])
            input_lengths, perm_idx = input_lengths.sort(0,descending=True)
            text = text[perm_idx]
            label = label[perm_idx]

            text = text.to(args.device)
            label = label.to(args.device)
            with torch.no_grad():
                output, _ = model(text, input_lengths)
                label = label.squeeze()
                loss = criterion(output, label)
            valid_losses.append(loss.item())
            total += label.size(0)
            valid_acc += acc(output, label)
        
        epoch_valid_loss = np.mean(valid_losses)
        epoch_valid_acc = valid_acc/total
        print('-------------------------------------------------------')
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))
        print(f'valid_loss : {epoch_valid_loss}')
        print('valid_accuracy : {:.3f}'.format(epoch_valid_acc*100))
        print('-------------------------------------------------------')

        # Plot to tensorboard
        tensorboard_plt.loss_plot('loss','valid',epoch_valid_loss,epoch)
        tensorboard_plt.loss_plot('accuracy','valid',epoch_valid_acc,epoch)
        tensorboard_plt.overlap_plot('loss',{'train':epoch_train_loss,'valid':epoch_valid_loss},epoch)
        tensorboard_plt.overlap_plot('accuracy',{'train':epoch_train_acc,'valid':epoch_valid_acc},epoch)


        # Save Model
        if epoch_valid_loss < min_loss:
            torch.save(model.state_dict(), args.model_name+".pt")
            print('Valid loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_loss, epoch_valid_loss))
            min_loss = epoch_valid_loss



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument('--vocab_size', type=int, default=60000, help="maximum vocab size")
    parser.add_argument('--batch_first', action='store_true', default=True,help="If true, then the model returns the batch first")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs to train for (default: 5)")
    parser.add_argument('--model_name', type=str, default='LSTM', help="Model name (default: LSTM)")
    parser.add_argument('--log_dir', type=str, default='./log/')
    
    args = parser.parse_args()

    """
    TODO: Build your model Parameters. You can change the model architecture and hyperparameters as you wish.
            (e.g. change epochs, vocab_size, hidden_dim etc.)
    """
    # Model hyperparameters
    input_size = args.vocab_size
    output_size = 4     # num of classes
    embedding_dim = 300 # embedding dimension
    hidden_dim = 64  # hidden size of RNN
    num_layers = 3

    # fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    # Make Train Loader
    train_dataset = TextDataset(args.data_dir, 'train_split', args.vocab_size)
    args.pad_idx = train_dataset.sentences_vocab.wtoi['<PAD>']
    train_loader = make_data_loader(train_dataset, args.batch_size, args.batch_first, shuffle=True)

    # Make valid Loader
    test_dataset = TextDataset(args.data_dir, 'valid_split', args.vocab_size)
    args.pad_idx = test_dataset.sentences_vocab.wtoi['<PAD>']
    test_loader = make_data_loader(test_dataset, args.batch_size, args.batch_first, shuffle=False)

    sentences_vocab = Vocabulary(args.vocab_size)
    sentences_vocab.load_vocabulary('data','./pickles')
    args.pad_idx = sentences_vocab.wtoi['<PAD>']
    args.unk_idx = sentences_vocab.wtoi['<UNK>']
    

    glove = GloVe(name='6B', dim=300)
    emb_init_weight = torch.zeros(args.vocab_size, 300)

    for word, idx in sentences_vocab.wtoi.items():
        if word in glove.stoi:
            emb_init_weight[idx] = glove.vectors[glove.stoi[word]]
        else:
            print(word + " not in glove")
            emb_init_weight[idx] = torch.randn(300)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("device : ", device)

    # instantiate model
    # model = BaseModel(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    # model = BILSTM(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    model = LSTM_ATTENTION(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first, device= args.device)
    model = model.to(device)

    # Training The Model
    train(args, train_loader,test_loader,model,emb_init_weight)