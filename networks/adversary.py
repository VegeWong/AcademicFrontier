import torch
import torchvision
from networks import resnet

class Adversary(object):
    '''
        A wrapper for adversary and different attack method
        Current: fgsm, ifgsm
        TO-Do: DeepFool, Carlini-Wagner's L2 attack
    '''

    def __init__(self, criterion, args):
        super(Adversary, self).__init__()

        self.net = resnet.resnet50(pretrained=args.pretrained, args=args)
        self.net.fc = torch.nn.Linear(2048, args.num_classes)
        self.criterion = criterion
        self.method = getattr(self, args.method)


    def generate(self, x, y, targeted, args):
        x.requires_grad = True
        self.net.zero_grad()
        
        return self.method(x, y, targeted, args)

    def none(self, x, y, targeted, args):
        self.net.eval()
        return x

    def fgsm(self, x, y, targeted, args):
        self.net.eval()

        # raw data
        pred = self.net(x)
        loss = self.criterion(pred, y)
        # clear grad
        self.net.zero_grad()
        if x.grad is not None:
            x.grad.fill_(0)
        loss.backward()

        # perturbation
        if targeted:
            xm = x - args.eps*x.grad.sign()
        else:
            xm = x + args.eps*x.grad.sign()

        # truncate
        xm = torch.clamp(xm, -1, 1)

        return xm

    def ifgsm(self, x, y, targeted, args):
        self.net.eval()
        alpha = args.eps / args.iterations

        for i in range(args.iterations):
            # print(i)
            # raw data
            pred = self.net(x)
            loss = self.criterion(pred, y)
            # clear grad
            self.net.zero_grad()
            if x.grad is not None:
                x.grad.fill_(0)
            loss.backward()

            # perturbation
            if targeted:
                x = x - alpha*x.grad.sign()
            else:
                x = x + alpha*x.grad.sign()

            # truncate
            x = torch.clamp(x, -1, 1)
            x = x.clone().detach().requires_grad_(True)
        return x

    # def patch(self, x, y, targeted, args):
