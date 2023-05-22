import os
import time
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler

from ddp_model import ConvNet


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0,1')
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

    os.environ['MASTER_ADDR'] = 'localhost'  # 0号机器的IP
    os.environ['MASTER_PORT'] = '19198'  # 0号机器的可用端口
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用哪些GPU
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    return args


def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')


def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def train(model, train_dloader, criterion, optimizer, scaler):
    model.train()
    for images, labels in train_dloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def test(model, test_dloader):
    local_rank = dist.get_rank()
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
    dist.reduce(size, 0, op=dist.ReduceOp.SUM)
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    if local_rank == 0:
        acc = correct / size
        print(f'Accuracy is {acc:.2%}')


def main(local_rank, args):
    init_ddp(local_rank)
    model = ConvNet().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    scaler = GradScaler()
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    g = get_ddp_generator()
    train_dloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True,
                                                sampler=train_sampler,
                                                generator=g)
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset)
    test_dloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=test_sampler)
    for epoch in range(args.epochs):
        if local_rank == 0:
            print(f'begin training of epoch {epoch + 1}/{args.epochs}')
        train_dloader.sampler.set_epoch(epoch)
        train(model, train_dloader, criterion, optimizer, scaler)
    if local_rank == 0:
        print(f'begin testing')
    test(model, test_dloader)
    dist.destroy_process_group()


if __name__ == '__main__':
    args = prepare()
    time_start = time.time()
    mp.spawn(main, args=(args, ), nprocs=torch.cuda.device_count())
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')
