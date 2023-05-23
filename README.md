# 一看就懂的DDP代码实践

本文对 PyTorch 中的 DistributedDataParallel（DDP）及混合精度模块的使用方式进行讲解。

关于 DDP 的原理及其相较于 DataParallel（DP）的优势，前人之述备矣，本文不再花费大量篇幅。
- [一看就懂的DDP代码实践](#一看就懂的ddp代码实践)
  - [原型](#原型)
    - [入口](#入口)
    - [初始化](#初始化)
    - [main](#main)
    - [模型](#模型)
    - [训练](#训练)
    - [测试](#测试)
  - [DDP示例](#ddp示例)
    - [入口](#入口-1)
    - [初始化](#初始化-1)
    - [main](#main-1)
      - [DDP初始化](#ddp初始化)
      - [模型](#模型-1)
      - [scaler](#scaler)
    - [训练](#训练-1)
    - [测试](#测试-1)
  - [Checklist](#checklist)
  - [PS](#ps)

本文参考了大量知乎文章、PyTorch文档及ChatGPT的回答，最主要参考的是[这篇文章](https://github.com/KaiiZhang/DDP-Tutorial/blob/main/DDP-Tutorial.md)。在这些基础上，结合个人项目中使用的情况，追求给出一个贴近 DL 项目现实且简洁、高效、可拓展的示例。

本文的代码针对单机多卡的情况，使用 nccl 后端，并通过 env 进行初始化。全部代码在[这里](https://github.com/rickyang1114/DDP-practice)。带有注释的行将着重讲解。

## 原型

首先，给出不使用 DDP 和 混合精度加速的代码。

### 入口

看看程序的入口：执行`main`函数并计时。

```python
if __name__ == '__main__':
    args = prepare()  ###
    time_start = time.time()
    main(args)
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')
```

### 初始化

第二行的`prepare`函数用于获取命令行参数：

```python
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
```

### main

在`main`函数中，首先通过`parse_args`获得一些训练相关的命令行参数，然后设定模型、损失函数、优化器、数据集。接着依次进行训练、测试，并保存模型的`state_dict`。

```python
def main(args):
    model = ConvNet().cuda()  ###
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
        num_workers=0,
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
        train(model, train_dloader, criterion, optimizer)  ###
    print(f'begin testing')
    test(model, test_dloader)  ###
    torch.save({'model': model.state_dict()}, 'origin_checkpoint.pt')
```

### 模型

上面第一行使用的模型为一个简单的 CNN：

```python
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
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
```

### 训练

训练使用的`train`函数：

```python
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
```

### 测试

测试使用的`test`函数：

```python
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
```

最后，启动命令如下：

```bash
python origin_main.py --gpu 0
```

输出的结果：

```bash
begin training of epoch 1/3
begin training of epoch 2/3
begin training of epoch 3/3
begin testing
Accuracy is 91.55%

time elapsed: 22.72 seconds
```

## DDP示例

在介绍完原型后，以下对代码进行改造，以使用DDP。

### 入口

首先，我们在`if __name__ == '__main__'`中启动 DDP：

```python
if __name__ == '__main__':
    args = prepare()  ###
    time_start = time.time()
    mp.spawn(main, args=(args, ), nprocs=torch.cuda.device_count())  #import torch.multiprocessing as mp
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')
```

`spawn`函数的主要参数包括以下几个：

1. `fn`，即上面传入的`main`函数。每个线程将执行一次该函数
2. `args`，即`fn`所需的参数。传给`fn`的参数必须写成元组的形式，哪怕像上面一样只有一个
3. `nprocs`启动的进程数，将其设置为`world_size`即可。不传默认为1，与`world_size`不一致会导致进程等待同步而一直停滞。

### 初始化

在`prepare`函数里面，也进行了一些 DDP 的配置：

```python
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
    
	# 下面几行是新加的，用于启动多进程 DDP
    os.environ['MASTER_ADDR'] = 'localhost'  # 0号机器的 IP
    os.environ['MASTER_PORT'] = '19198'  # 0号机器的可用端口，随便选一个没被占用的
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用哪些 GPU
    world_size = torch.cuda.device_count() # 就是上一行使用的 GPU 数量
    os.environ['WORLD_SIZE'] = str(world_size)
    return args
```

### main

再来看看`main`函数里面添加了什么。首先是其添加一个额外的参数`local_rank`（在`mp.spawn`里面不用传，会自动分配）

```python
def main(local_rank, args):
    init_ddp(local_rank)  ### 进程初始化
    model = ConvNet().cuda()  ### 模型的 forward 方法变了
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  ### 转换模型的 BN 层
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  ### 套 DDP
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    scaler = GradScaler()  ###  用于混合精度训练
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)  ### 用于在 DDP 环境下采样
    g = get_ddp_generator()  ###
    train_dloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True,
                                                sampler=train_sampler,
                                                generator=g)  ### 添加额外的 generator
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset)  ### 用于在 DDP 环境下采样
    test_dloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=test_sampler)
    for epoch in range(args.epochs):
        if local_rank == 0:  ### 防止每个进程都输出一次
            print(f'begin training of epoch {epoch + 1}/{args.epochs}')
        train_dloader.sampler.set_epoch(epoch)  ### 防止采样出 bug
        train(model, train_dloader, criterion, optimizer, scaler)
    if local_rank == 0:
        print(f'begin testing')
    test(model, test_dloader)
    if local_rank == 0:  ### 防止每个进程都保存一次
        torch.save({'model': model.state_dict(), 'scaler': scaler.state_dict()}, 'ddp_checkpoint.pt')
    dist.destroy_process_group()  ### 最后摧毁进程，和 init_process_group 相对
```

#### DDP初始化

首先，根据用`init_ddp`函数对模型进行初始化。这里我们使用 nccl 后端，并用 env 作为初始化方法：

```python
def init_ddp(local_rank):
    # 有了这一句之后，在转换device的时候直接使用 a=a.cuda()即可，否则要用a=a.cuda(local+rank)
    torch.cuda.set_device(local_rank)  
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
```

在完成了该初始化后，可以很轻松地在需要时获得`local_rank`、`world_size`，而不需要作为额外参数从`main`中一层一层往下传。

```python
import torch.distributed as dist
local_rank = dist.get_rank()
world_size = dist.get_world_size()
```

比如需要`print`, `log`, `save_state_dict`时，由于多个进程拥有相同的副本，故只需要一个进程执行即可，比如：

```python
if local_rank == 0:
    print(f'begin testing')
if local_rank == 0:  ### 防止每个进程都保存一次
    torch.save({'model': model.state_dict(), 'scaler': scaler.state_dict()}, 'ddp_checkpoint.pt')
```

#### 模型

为了加速推理，我们在模型的`forward`方法里套一个`torch.cuda.amp.autocast()`：

使得`forward`函数变为：

```python
def forward(self, x):
    with torch.cuda.amp.autocast():  # 混合精度，加速推理
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
    return out
```

autocast 也可以在推理的时候再套，但是在这里套最方便，而且适用于所有情况。

在模型改变之后，使用`convert_sync_batchnorm`和`DistributedDataParallel`对模型进行包装。

#### scaler

创建 scaler，用于训练时对 loss 进行 scale：

```python
from torch.cuda.amp import GradScaler
scaler = GradScaler()  ###  用于混合精度训练
```

### 训练

训练时，需要使用 DDP 的sampler，并且在`num_workers > 1`时需要传入`generator`，否则对于同一个worker，所有进程将会获得同样的data，参见[这篇文章](https://zhuanlan.zhihu.com/p/618639620)。

```python
def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)  ### 用于在 DDP 环境下采样
g = get_ddp_generator()  ###
train_dloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,  ### shuffle 通过 sampler 完成
                                            num_workers=4,
                                            pin_memory=True,
                                            sampler=train_sampler,
                                            generator=g)  ### 添加额外的 generator
```

并且在多个`epoch`的训练时，需要设置`train_dloader.sampler.set_epoch(epoch)`。

下面来看看`train`函数。

```python
def train(model, train_dloader, criterion, optimizer, scaler):
    model.train()
    for images, labels in train_dloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()  ###
        scaler.step(optimizer)  ###
        scaler.update()  ###
```

最后三行发生了改变。相较于原始的`loss.backward`、`optimizer.step()`，这里通过`scaler`对梯度进行缩放，防止由于使用混合精度导致损失下溢，并且对`scaler`自身的状态进行更新呢。如果有多个`loss`，它们也使用同一个`scaler`。如果需要保存模型的`state_dict`并且在后续继续训练（比如预训练-微调模式），最好连带`scaler`的状态一起保留，并在后续的微调过程中和模型的参数异同加载。

### 测试

测试时，需要将多个进程的数据`reduce`到一张卡上。注意，在`test`函数的外面加上`if local_rank == 0`，否则多个进程会彼此等待而陷入死锁。

```python
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
    dist.reduce(size, 0, op=dist.ReduceOp.SUM)  ###
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)  ###
    if local_rank == 0:
        acc = correct / size
        print(f'Accuracy is {acc:.2%}')
```

注释的两行即为所需添加的`reduce`操作。

至此，添加的代码讲解完毕。

启动的方式变化不大：

```bash
python ddp_main.py --gpu 0,1
```

相应的结果：

```bash
begin training of epoch 1/3
begin training of epoch 2/3
begin training of epoch 3/3
begin testing
Accuracy is 89.21%

time elapsed: 30.82 seconds
```

## Checklist

在写完 DDP 的代码之后，最好检查一遍，否则很容易因为漏了什么而出现莫名奇妙的错误，比如程序卡着不动了，也不报错）

大致需要检查：

1. DDP 初始化有没有完成，包括`if __name__ == '__main__'`里和`main`函数里的。退出`main`函数时摧毁进程。
2. 模型的封装，包括autocast，BN 层的转化和 DDP 封装
3. 指定`train_dloader`的`sampler`、`generator`和`shuffle`，并且在每个`epoch`设置`sampler`，测试集、验证集同理。
4. 训练时使用`scaler`对`loss`进行`scale`
5. 对于`print`、`log`、`save`等操作，仅在一个线程上进行。
6. 测试时进行`reduce`

## PS

多个线程大致相当于增大了相应倍数的`batch_size`，最好相应地调一调`batch_size`和学习率。本文没有进行调节，导致测试获得的准确率有一些差别。

模型较小时速度差别不大，反而DDP与混合精度可能因为一些初始化和精度转换耗费额外时间而更慢。在模型较大时，DDP + 混合精度的速度要明显高于常规，且能降低显存占用。
