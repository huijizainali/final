'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from Resnet import ResNet34
from utils import progress_bar
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from VIT import ViT
import time


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#val_percent = 0.2
trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
classes = trainset.classes

#n_val = int(len(trainset) * val_percent)
#n_train = len(trainset) - n_val
#trainset, valset = torch.utils.data.random_split(trainset, [n_train, n_val])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
#valloader = torch.utils.data.DataLoader(
 #   valset, batch_size=128, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)



# Model
print('==> Building model..')
#net = VGG('VGG19')
#net = ResNet34()
#summary(net,(3,32,32),batch_size=-1)
#resnet0 = timm.create_model('resnet_small_resnet26d_224', pretrained=False,num_classes=100)
net =ViT(image_size = 32,patch_size = 4,num_classes = 100,dim = 256,depth = 8,
        heads = 8,mlp_dim = 128,dropout = 0.1,emb_dropout = 0.1)
#summary(net,(3,32,32),batch_size=-1)
net = net.to(device)
#summary(net,(3,32,32),batch_size=-1)


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
writer = SummaryWriter('./path/vit2/log')
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        writer.add_scalar('Loss_train_vit',loss, epoch)  # tensorboard train loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            writer.add_scalar('Loss_test_vit',loss, epoch)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    writer.add_scalar('Acc_test_vit', acc, epoch)  # tensorboard acc
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/vit.pth')
        best_acc = acc


def main():
    
    #val_loss_list = []
    #val_acc = []
    
    for epoch in range(start_epoch, start_epoch+500):
        train(epoch)
       # val(epoch, val_loss_list, val_acc)
        test(epoch)
        scheduler.step()
        
    #torch.save(val_loss_list, 'val_loss_list')
   # torch.save(val_acc, 'val_acc')

if __name__=='__main__':
    main()
    