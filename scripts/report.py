import os
import sys
import argparse
# import math
from PIL import Image
from PIL import ImageChops
import numpy as np

_default_save_rootdir = './reports/'
_default_model_rootdir = './models/'

parser = argparse.ArgumentParser(description="ResultWrapper Configuration")
parser.add_argument('--read_path', type=str, default=None)
parser.add_argument('--save_path', type=str, default=None)


class ResultWrapper(object):
    def __init__(self, read_path=None, save_path=None):
        if read_path is None:
            raise TypeError('The report object can not be NoneType')
        self.read_path = read_path

        par_dir, cur_dir = os.path.split(read_path)
        cur_dir = str(filter(lambda ch: ch not in '0123456789-:', cur_dir))
        
        if save_path is None:
            self.save_path = _default_save_rootdir + cur_dir
        else:
            self.save_path = save_path

        self.model_path = _default_model_rootdir + cur_dir

    def generate(self):
        for item in os.listdir(self.read_path):
            temp_path = self.read_path + '/' + item
            print(temp_path)
            if os.path.isdir(temp_path):
                img_origin = Image.open(temp_path + '/' + 'origin.JPEG')
                img_modified = Image.open(temp_path + '/' + 'modified.JPEG')
                img_delta = ImageChops.difference(img_modified, img_origin)
                img_modified.show()
                img_origin.show()
                # print(np.asarray(img_delta))
                break
                img_delta.save(temp_path + '/' + "difference.JPEG")

def main(args):
    obj = ResultWrapper(read_path=args.read_path, save_path=args.save_path)
    obj.generate()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.read_path is None:
        raise argparse.ArgumentError('Read path can not be NoneType')
    main(args)
            