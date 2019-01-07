import argparse


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