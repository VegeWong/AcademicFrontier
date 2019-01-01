import torch
import torchvision
from networks import resnet
class Adversary():
    '''
        A wrapper for adversary and different attack method
        Current: fgsm, ifgsm
        TO-Do: DeepFool, Carlini-Wagner's L2 attack
    '''

    def __init__(self, criterion, args):
        super(Adversary, self).__init__()

        self.net = resnet.resnet34(num_classes=args.num_classes, pretrained=args.pretrained)
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

        # truncate if the value range is limited
        if args.truncate:
            xm = torch.clamp(xm, args.vmin, args.vmax)

        return xm

    def ifgsm(self, x, y, targeted, args):
        self.net.eval()
        alpha = args.eps / args.iterations

        for i in range(args.iterations):
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

            # truncate if the value range is limited
            if args.truncate:
                x = torch.clamp(x, args.vmin, args.vmax)

        return x

