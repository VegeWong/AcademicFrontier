import argparse

def str2bool(param):
    if param.lower() in ('yes', 'true', 't', '1'):
        return True
    elif param.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unknown Value')