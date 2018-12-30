import os
import networks
import argparse
import datetime
import torch
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from utils import str2bool

parser = argparse.ArgumentParser(description="Model Configuration")
parser.add_argument('--mode', type=str, default='train',
                    help='select the mode of adversary(advesary train or attack)')
parser.add_argument('--method', type=str, default='none',
                    help='select the attack method of adversary')
parser.add_argument('--iterations', type=int, default=1,
                    help='determine the number of iterations of ifgsm')
parser.add_argument('--datadir', type=str, default='./data',
                    help='set the directory of training/test data')
parser.add_argument('--resultdir', type=str, default='./results',
                    help='output dir')
parser.add_argument('--model', type=str, default='./models/001.model',
                    help='directory for loading models')
parser.add_argument('--name', type=str, default=datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S'))
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('--eps', type=float, default=0.01,
                    help='epsilon')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--transform', type=str2bool, default=False)
parser.add_argument('--pretrained', type=str2bool, default=False,
                    help='Use pretrained resnet to accelerate convergence')
parser.add_argument('--cuda', type=str2bool, default=False,
                    help='Use GPU')
parser.add_argument('--gpu_id', type=int, default=3,
                    help='GPU ID used')                   

criterion = torch.nn.CrossEntropyLoss()

def main(args):
    print(args)

    # Use existing model or pretrained resnet
    net = networks.resnet.resnet34(pretrained=args.pretrained)
    if os.path.exists(args.model):
        net.load_state_dict(torch.load(args.model))
    
    adv = networks.adversary.Adversary(net, criterion, args)

    if args.mode == 'train':
        train(net, adv, args)
    elif args.mode == 'generate':
        generate(net, adv, args)
    else:
        raise argparse.ArgumentError("Unknown value for --mode: "+args.mode)


def train(net, adv, **kargs):
    train_set = datasets.ImageFolder(args.datadir+'train/')
    train_transform = transforms.Compose()
    if kargs.transform:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(size=kargs['image_size']),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    else:
        train_transform = transforms.Compose([transforms.Resize(kargs['image_size']),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    train_loader = DataLoader(dataset=train_set,
                                batch_size=args.batch_size,
                                transform = train_transform,
                                Shuffle=False)
    
    

def generate(net, adv, **kargs):



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)