import torch
import torchvision

class Adversary(object):
    '''
        A wrapper for adversary and different attack method
        Current: fgsm, ifgsm
        TO-Do: DeepFool, Carlini-Wagner's L2 attack
    '''

    def __init__(self, net, criterion, **kargs):
        super(Adversary, self).__init__()

        self.net = net
        self.criterion = criterion
        self.method = getattr(self, kargs['method'])


    def generate(self, x, y, targeted, **kargs):
        x.requires_grad = True
        self.net.zero_grad()
        
        return self.method(x, y, targeted, kargs)

    def none(self, x, y, targeted, **kargs):
        self.net.eval()
        return x, self.net(x)

    def fgsm(self, x, y, targeted, **kargs):
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
            xm = x - kargs['eps']*x.grad.sign()
        else:
            xm = x + kargs['eps']*x.grad.sign()

        # truncate if the value range is limited
        if kargs['truncate']:
            xm = torch.clamp(xm, kargs['vmin'], kargs['vmax'])

        ym = self.net(xm)
        return xm, ym

    def ifgsm(self, x, y, targeted, **kargs):
        self.net.eval()
        alpha = kargs['eps'] / kargs['iterations']

        for i in range(kargs['iterations']):
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
            if kargs['truncate']:
                x = torch.clamp(xm, kargs['vmin'], kargs['vmax'])

        return x, self.net(x)

