import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import Dataset
import vanilla_mod as net


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('-data_options', type=int, required=True, help='Choosing training set, 0 for CIFAR10, 1 for '
                                                                       'CIFAR100')
    parser.add_argument('-b', type=int, default=10,
                        help='batch size')
    parser.add_argument('-model_select', type=int, default=0,
                        help='choose 0 for vanilla model and 1 for upgraded model')
    parser.add_argument('-l', type=str, default='encoder.pth',
                        help='load encoder')
    parser.add_argument('-s', type=str, default='decoder.pth',
                        help='save decoder')

    args = parser.parse_args()
    device = torch.device("cpu")
    print(f"Device being used: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.data_options == 0:
        classes = 10
        testSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=args.b, shuffle=False, num_workers=2)

    else:
        classes = 100
        testSet = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=args.b, shuffle=False, num_workers=2)

    encoder = net.vgg
    encoder.load_state_dict(torch.load(args.l, map_location=device))
    encoder = encoder.to(device)

    if args.model_select == 0:
        from vanilla_mod import ClassificationModel

        model = ClassificationModel(encoder, classes).to(device)
    else:
        from upgraded_mod import DenseResNet152
        model = DenseResNet152(encoder, classes).to(device)

    model.load_state_dict(torch.load(args.s, map_location=device))
    model.eval()

    top1_accuracy = 0.0
    top5_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, data in enumerate(testLoader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_accuracy += acc1[0]
            top5_accuracy += acc5[0]
            num_batches += 1

    top1_avg_accuracy = top1_accuracy / num_batches
    top5_avg_accuracy = top5_accuracy / num_batches

    print(f'Top 1 accuracy: {top1_avg_accuracy:.2f}%, Top 5 accuracy: {top5_avg_accuracy:.2f}%')