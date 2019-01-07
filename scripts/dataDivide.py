import random
import os
import sys

image_dir = '/home/wanghanjing/AcademicFrontier/data/'
train_dir = '/home/wanghanjing/AcademicFrontier/data/train/'
test_dir = '/home/wanghanjing/AcademicFrontier/data/test/'
val_dir = '/home/wanghanjing/AcademicFrontier/data/val/'
def main():
    for folder in os.listdir(image_dir):
        if folder != 'train' and folder != 'test' and folder != 'val':
            os.popen('mkdir '+val_dir+folder)
            os.popen('mkdir '+train_dir+folder)
            os.popen('mkdir '+test_dir+folder)
            contents = os.listdir(image_dir+folder)
            rs = random.sample(range(len(contents)), int(0.2*len(contents)))
            val = random.sample(range(len(rs)), int(0.4*len(rs)))
            test_mask = [contents[i] for i in rs]
            val_mask = [test_mask[i] for i in val]
            for img in val_mask:
                os.popen('mv '+image_dir+folder+'/'+img+' '+val_dir+folder+'/'+img)
            for img in test_mask:
                os.popen('mv '+image_dir+folder+'/'+img+' '+test_dir+folder+'/'+img)
            os.popen('mv '+image_dir+folder+'/*'+' '+train_dir+folder+'/')

if __name__ == '__main__':
    main()
