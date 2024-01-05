import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torchvision import transforms
import vanilla_mod as net


def plot_loss(loss_list, save_path):
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification")
    parser.add_argument('-data_options', type=int, required=True, help='Choosing training set, 0 for CIFAR10, 1 for '
                                                                       'CIFAR100')
    parser.add_argument('-e', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('-lr', type=int, default=1e-3,
                        help='learning rate')
    parser.add_argument('-lr_decay', type=int, default=5e-5,
                        help='learning rate decay')
    parser.add_argument('-b', type=int, default=10,
                        help='batch size')
    parser.add_argument('-model_select', type=int, default=0,
                        help='choose 0 for vanilla model and 1 for upgraded model')
    parser.add_argument('-l', type=str, default='encoder.pth',
                        help='load encoder')
    parser.add_argument('-s', type=str, default='decoder.pth',
                        help='save decoder')
    parser.add_argument('-p', type=str, default='decoder.png',
                        help='Value for p')
    parser.add_argument('-cuda', type=str, help='cuda', default='Y')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")

    print("Args:", args)

    print("Cuda avaliable: ", torch.cuda.is_available())

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.data_options == 1:
        print('CIFAR100 Dataset')
        classes = 100
        trainSet = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=args.b, shuffle=True, num_workers=4)
        valSet = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        valLoader = torch.utils.data.DataLoader(valSet, batch_size=args.b, shuffle=False, num_workers=4)

    else:
        print('CIFAR10 Dataset')
        classes = 10
        trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=args.b, shuffle=True, num_workers=4)
        valSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        valLoader = torch.utils.data.DataLoader(valSet, batch_size=args.b, shuffle=False, num_workers=4)

    encoder = net.vgg
    encoder.load_state_dict(torch.load(args.l, map_location=device))
    encoder = encoder.to(device)

    if args.model_select == 0:
        from vanilla_mod import ClassificationModel

        model = ClassificationModel(encoder, classes).to(device)
    else:
        from upgraded_mod import DenseResNet152

        model = DenseResNet152(encoder, classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for optimizer_ in optimizer.param_groups:
        optimizer_['lr'] = args.lr

    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)

    losses = []
    best_loss = float('inf')
    for epoch in range(args.e):
        print(f'Epoch {epoch + 1}')
        epoch_start = time.time()
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainLoader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        epoch_end = time.time()

        avg_loss = running_loss / len(trainLoader)
        losses.append(avg_loss)
        print(f'Training loss: {avg_loss:.3f}, Time taken: {epoch_end - epoch_start:.3f} seconds')

        valTime = time.time()
        valLoss = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(valLoader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valLoss += loss.item()
        valLoss /= len(valLoader)

        valTime = time.time() - valTime

        print(f'Validation loss: {valLoss:.3f}, Time taken: {valTime:.3f} seconds')
        if valLoss < best_loss:
            print('Saving model')
            best_loss = valLoss
            torch.save(model.state_dict(), args.s)
        print()

    print('Finished Training')
    plot_loss(losses, args.p)
