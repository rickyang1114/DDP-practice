import time
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        with torch.cuda.amp.autocast():  # 混合精度，加速推理
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
        return out


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('-e',
                        '--epochs',
                        default=3,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        metavar='N',
                        help='number of batchsize')
    args = parser.parse_args()
    return args


def train(model, train_dloader, criterion, optimizer):
    model.train()
    for images, labels in train_dloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, test_dloader):
    model.eval()
    size = torch.tensor(0.).cuda()
    correct = torch.tensor(0.).cuda()
    for images, labels in test_dloader:
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(images)
            size += images.size(0)
        correct += (outputs.argmax(1) == labels).type(torch.float).sum()
    acc = correct / size
    print(f'Accuracy is {acc:.2%}')


def main(args):
    model = ConvNet().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_dloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)
    test_dloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    for epoch in range(args.epochs):
        print(f'begin training of epoch {epoch + 1}/{args.epochs}')
        train(model, train_dloader, criterion, optimizer)
    print(f'begin testing')
    test(model, test_dloader)
    torch.save({'model': model.state_dict()}, 'origin_checkpoint.pt')


if __name__ == '__main__':
    args = prepare()
    time_start = time.time()
    main(args)
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')
