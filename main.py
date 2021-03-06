import os
import networks
import argparse
import datetime
from PIL import Image

import torch
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils

parser = argparse.ArgumentParser(description="Model Configuration")
parser.add_argument('--mode', type=str, default='train',
                    help='select the mode of adversary(advesary train or attack)')
parser.add_argument('--method', type=str, default='none',
                    help='select the attack method of adversary')
parser.add_argument('--iterations', type=int, default=10,
                    help='determine the number of iterations of ifgsm')
parser.add_argument('--datadir', type=str, default='./data',
                    help='set the directory of training/test data')
parser.add_argument('--resultdir', type=str, default='./models/',
                    help='output dir')
parser.add_argument('--generatedir', type=str, default='./images/',
                    help='directory for saving generated images')
parser.add_argument('--dropout', type=utils.str2bool, default=True,
                    help='always dropout in the units of network')
parser.add_argument('--stochastic', type=utils.str2bool, default=True,
                    help='stochastic depth')
parser.add_argument('--model', type=str, default='./result/model.pkl',
                    help='directory for loading models')
parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('--seed', type=float, default=7,
                    help='torch random seed')
parser.add_argument('--eps', type=float, default=0.1,
                    help='epsilon')
parser.add_argument('--dropout_prob', type=float, default=0.1,
                    help='epsilon')
parser.add_argument('--num_classes', type=int, default=10,
                    help='number of classeses')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--num_decay', type=int, default=5)
parser.add_argument('--decay', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--image_size', type=int, default=112)
parser.add_argument('--transform', type=utils.str2bool, default=False)
parser.add_argument('--pretrained', type=utils.str2bool, default=True,
                    help='Use pretrained resnet to accelerate convergence')
parser.add_argument('--cuda', type=utils.str2bool, default=True,
                    help='Use GPU')
parser.add_argument('--gpu_id', type=int, default=3,
                    help='GPU ID used')                   

criterion = torch.nn.CrossEntropyLoss()

def main(args):
    utils.printArgs(args)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_id)

    # Use existing model or pretrained resnet
    union_net = networks.adversary.Adversary(criterion, args)
    if os.path.exists(args.model):
        union_net.net.load_state_dict(torch.load(args.model))

    if args.mode == 'train':
        train(union_net, args)
    elif args.mode == 'generate':
        generate(union_net, args)
    else:
        raise argparse.ArgumentError("Unknown value for --mode: "+args.mode)


def train(union_net, args):
    # saving arg settings 
    os.popen('mkdir '+args.resultdir+args.name)
    os.popen('touch '+args.resultdir+args.name+'/args.txt')
    f = open(args.resultdir+args.name+'/args.txt', 'w')
    utils.printArgs(args, outputFile=f)

    optimizer = torch.optim.Adam(union_net.net.parameters(), lr=args.lr)
    train_transform = transforms.ToTensor()
    if args.transform:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
                                ])
    else:
        train_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
                                ])
    train_set = datasets.ImageFolder(args.datadir+'/train/', transform=train_transform)
    val_set = datasets.ImageFolder(args.datadir+'/val/', transform=train_transform)
    train_loader = DataLoader(dataset=train_set,
                                batch_size=args.batch_size,
                                shuffle=True)
    val_loader = DataLoader(dataset=val_set,
                                batch_size=32,
                                shuffle=True)
    
    if args.cuda:
        union_net.net = union_net.net.cuda()

    for epoch_counter in range(args.epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0
        if (epoch_counter + args.num_decay == args.epochs - 1):
            optimizer.param_groups[0]['lr'] *= args.decay

        for batch_index, (data, label) in enumerate(train_loader):
            if args.cuda:
                data = data.cuda()
                label = label.cuda()

            # let the batches with even indexes perturbed by the adversary
            if batch_index % 2 == 0:
                data = union_net.generate(data, label, False, args)

            # switch to training mode
            union_net.net.train()
            pred = union_net.net(data)
            loss = union_net.criterion(pred, label)
            epoch_train_loss += loss.item()
            
            union_net.net.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_train_loss /= batch_index + 1
        for batch_index, (data, label) in enumerate(val_loader):
            if args.cuda:
                data = data.cuda()
                label = label.cuda()
            # let the batches with even indexes perturbed by the adversary
            if batch_index % 2 == 0:
                data = union_net.generate(data, label, False, args)

            # switch to training mode
            union_net.net.eval()
            pred = union_net.net(data)
            loss = union_net.criterion(pred, label)
            epoch_val_loss += loss.item()

        epoch_val_loss /= batch_index + 1
        print('Average loss of epoch %d:( train set:%.3f, validation set:%.3f )' %(epoch_counter, epoch_train_loss, epoch_val_loss))
        print('Average loss of epoch %d:( train set:%.3f, validation set:%.3f )' %(epoch_counter, epoch_train_loss, epoch_val_loss), file=f)
    f.close()
    torch.save(union_net.net.state_dict(), args.resultdir+args.name+'/state_dict.pkl')

def generate(union_net, args):
    folder_name = os.path.split(os.path.split(args.model)[0])[1] + args.method
    os.popen('mkdir '+args.generatedir+folder_name)
    os.popen('cp '+os.path.split(args.model)[0]+'/args.txt'+' '+args.generatedir+folder_name+'/ModelArgs.txt')
    os.popen('touch '+args.generatedir+folder_name+'/GenerateArgs.txt')
    
    test_transform = transforms.ToTensor()
    if args.transform:
        test_transform = transforms.Compose([transforms.RandomResizedCrop(size=args.image_size),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
                                ])
    else:
        test_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
                                ])
    test_set = datasets.ImageFolder(args.datadir+'/test/', transform=test_transform)
    test_loader = DataLoader(dataset=test_set,
                                batch_size=args.batch_size,
                                shuffle=True)
           
    if args.cuda:
        union_net.net = union_net.net.cuda()

    loss = 0.0
    tran = transforms.Compose([transforms.Normalize(std=[2, 2, 2], mean=[-1, -1, -1]),
                            transforms.ToPILImage()
                            ]) 
    for batch_index, (data, label) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
            label = label.cuda()

        datam = union_net.generate(data, label, False, args)

        pred = union_net.net(datam)
        loss += union_net.criterion(pred, label).item()
        # print(data)
        # d = data.detach().cpu()
        # dm = datam.detach().cpu()
        # pred = torch.argmax(pred, 1)
        # for i in range(len(data)):
        #     dirname = args.generatedir+folder_name+'/batch%dimage%d'%(batch_index, i)
        #     os.mkdir(dirname)
        #     tran(d[i]).save(dirname+'/origin.JPEG')
        #     tran(dm[i]).save(dirname+'/modified.JPEG')
        #     os.popen('touch '+dirname+'/Labels.txt')
        #     with open(dirname+'/Labels.txt', 'w') as f:
        #         print('origin:'+str(label[i]), file=f)
        #         print('modified:'+str(pred[i]), file=f)
        
    loss /= batch_index + 1
    print('Generating loss=%.3f' %(loss))
    with open(args.generatedir+folder_name+'/GenerateArgs.txt', 'w') as f:
        utils.printArgs(args, outputFile=f)
        print('Generating loss=%.3f' %(loss), file=f)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
