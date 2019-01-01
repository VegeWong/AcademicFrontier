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
parser.add_argument('--resultdir', type=str, default='./results/',
                    help='output dir')
parser.add_argument('--model', type=str, default='./models/001.model',
                    help='directory for loading models')
parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('--betas', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('--eps', type=float, default=0.01,
                    help='epsilon')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_size', type=int, default=112)
parser.add_argument('--transform', type=str2bool, default=False)
parser.add_argument('--pretrained', type=str2bool, default=False,
                    help='Use pretrained resnet to accelerate convergence')
parser.add_argument('--truncate', type=str2bool, default=True)
parser.add_argument('--vmin', type=float, default=-1.0)
parser.add_argument('--vmax', type=float, default=1.0)
parser.add_argument('--cuda', type=str2bool, default=False,
                    help='Use GPU')
parser.add_argument('--gpu_id', type=int, default=3,
                    help='GPU ID used')                   

criterion = torch.nn.CrossEntropyLoss()

def main(args):
    print(args)
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
    optimizer = torch.optim.Adam(union_net.net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    train_transform = transforms.ToTensor()
    if args.transform:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(size=args.image_size),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    else:
        train_transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    train_set = datasets.ImageFolder(args.datadir+'train/', transform=train_transform)
    train_loader = DataLoader(dataset=train_set,
                                batch_size=args.batch_size,
                                Shuffle=False)
    
    if args.cuda:
        union_net = union_net.cuda()

    for epoch_counter in range(args.epochs):
        for batch_index, (data, labels) in enumerate(train_loader):
            if args.cuda:
                data = data.cuda()
                labels = labels.cuda()
            
            data_1, data_2 = torch.split(data, 2, dim=0)
            label_1, label_2 = torch.split(labels, 2, dim=0)
            data_2 = union_net.generate(data_2, label_2, False, args)
            data = torch.cat((data_1, data_2), 0)
            label = torch.cat((label_1, label_2), 0)

            union_net.net.train()
            pred = union_net.net(data)
            loss = union_net.criterion(pred, label)

            union_net.net.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(union_net.net.state_dict, args.resultdir+args.name+'.pkl')

def generate(net, adv, args):
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)