import argparse
import functools as ft
def printArgs(args, outputFile=None):
    print("---------------Args-----------------", file=outputFile)
    for arg in vars(args):
        print('%s:%s' %(arg, getattr(args, arg)), file=outputFile)
    print("------------------------------------", file=outputFile)

def str2bool(param):
    if param.lower() in ('yes', 'true', 't', '1'):
        return True
    elif param.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unknown Value')


def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def reduce_mean(x, keepdim=True):
    numel = ft.reduce(op.mul, x.size()[1:])
    x = reduce_sum(x, keepdim=keepdim)
    return x / numel


def reduce_min(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.min(a, keepdim=keepdim)[0]
    return x


def reduce_max(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.max(a, keepdim=keepdim)[0]
    return x


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5

def dist(a, b, n, keepdim=True):
    '''
    a: tensor 0,
    b: tensor 1,
    n: norm(l-2, l-0 l-inf)
    '''
    t = (a - b)**n
    return reduce_sum(t, keepdim=keepdim)