import argparse
from math import degrees
import plotter 
from model import *
from datasets import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def compute_accuracy(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    cost = 0
    n_batches = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            cost += loss.item()
            outputs.argmax()

            _, predicted = torch.max(outputs, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
            n_batches += 1
        cost /= n_batches

    return correct , total, cost

def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data', type=str, default='TotalDataset', help='Path to training data')
    parser.add_argument('--model', type=str, default='model.pt', help='Model to test')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = vars(parser.parse_args())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(17)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args['data'] == 'TotalDataset':
        test_data = TotalDataset(train=False, transform=data_transform)
    elif args['data'] == 'MNIST3D':
        test_data = MNIST3D(train=False, transform=data_transform)
    elif args['data'] == 'SVHN':
        test_data = SVHN(train=False, transform=data_transform)
    else :
        test_data = DigitData(train=False, transform=data_transform)

    test_data = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)

    print("Dataload Finished ")

    model = ResNet18(BasicBlock, [1,1,1,1]).to(device)
    model.load_state_dict(torch.load(args['model'], map_location=device))

    print("Model Loaded : ", args['model'])
    correct, total, cost = compute_accuracy(model, test_data, device)
    print("Accuracy : ", correct/total)
    print("Cost : ", cost)
    print("Total : ", total)
    print("Correct : ", correct)
            

if __name__ == '__main__':
    main()