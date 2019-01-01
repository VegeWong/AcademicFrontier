import random
import os
import sys

image_dir = '/home/wanghanjing/AcademicFrontier/data/'
train_dir = '/home/wanghanjing/AcademicFrontier/data/train/'
test_dir = '/home/wanghanjing/AcademicFrontier/data/test/'
sample_rate = 0.2
def main():
    for folder in os.listdir(image_dir):
        if folder != 'train' and folder != 'test':
            os.popen('mkdir '+train_dir+folder)
            os.popen('mkdir '+test_dir+folder)
            contents = os.listdir(image_dir+folder)
            length = len(contents)
            rs = random.sample(range(length), int(sample_rate*length))
            test_mask = [contents[i] for i in rs]
            for img in test_mask:
                os.popen('mv '+image_dir+folder+'/'+img+' '+test_dir+folder+'/'+img)
            os.popen('mv '+image_dir+folder+'/*'+' '+train_dir+folder+'/')

if __name__ == '__main__':
    main()
