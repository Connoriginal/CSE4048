import argparse
from math import degrees
from sched import scheduler
from datasets import *
import plotter 
from model import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data', type=str, default='TotalDataset', help='Path to training data')
    parser.add_argument('--load-model', default=None, help="Model's state_dict")
    parser.add_argument('--model', type=str, default='CNN', help='Model to train')
    parser.add_argument('--model_path', type=str, default='model.pt', help='Path to save model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = vars(parser.parse_args())
    
    tensorboard_plt = plotter.TensorboardPlotter('./log/' + args['model'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(17)

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if args['data'] == 'TotalDataset':
        train_data = TotalDataset(train=True, transform=data_transform)
        test_data = TotalDataset(train=False, transform=data_transform)
    elif args['data'] == 'MNIST3D':
        train_data = MNIST3D(train=True, transform=data_transform)
        test_data = MNIST3D(train=False, transform=data_transform)
    elif args['data'] == 'SVHN':
        train_data = SVHN(train=True, transform=data_transform)
        test_data = SVHN(train=False, transform=data_transform)
    else :
        train_data = DigitData(train=True, transform=data_transform)
        test_data = DigitData(train=False, transform=data_transform)

    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    test_data = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)

    print("Dataload Finished ")
    if args['model'] == 'CNN':
        model = RobustModel().to(device)
    else :
        model = ResNet18(BasicBlock, [1,1,1,1]).to(device)

    if args['load_model'] != None:
        model.load_state_dict(torch.load(args['load_model'], map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
    
    
    for epoch in range(args['epochs']):
        model.train()
        cost = 0
        n_batches = 0
        
        for i,(X,Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()
            
            cost += loss.item()
            n_batches += 1
        
        cost /= n_batches
        
        # Tensorboard
        tensorboard_plt.loss_plot('loss', 'train', cost, epoch)
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, cost))

        model.eval()
        val_cost = 0
        val_n_batches = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i,(X,Y) in enumerate(test_data):
                X = X.to(device)
                Y = Y.to(device)
                output = model(X)
                loss = criterion(output, Y)
                val_cost += loss.item()
                output.argmax()
                _, predicted = torch.max(output.data, 1)
                total += len(Y)
                correct += (predicted == Y).sum().item()
                val_n_batches += 1
            val_cost /= val_n_batches
            accuracy = correct / total
            tensorboard_plt.loss_plot('accuracy', 'val', accuracy, epoch)
            tensorboard_plt.loss_plot('loss', 'val', val_cost, epoch)
            tensorboard_plt.overlap_plot('loss',{'train':cost, 'val':val_cost}, epoch)
            print('Validation Loss: {:.4f}'.format(val_cost))
            print('Validation Accuracy: {:.4f}'.format(accuracy))
            print("--------------------------------------------------------------------------------")
            scheduler.step(val_cost)
        if(epoch % 5 == 0):
            torch.save(model.state_dict(), args['model_path'].split('.')[0] + '_' + str(epoch) + '.pt')

    
    torch.save(model.state_dict(), args['model_path'])

if __name__ == '__main__':
    main()