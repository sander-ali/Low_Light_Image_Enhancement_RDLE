import os
import sys
import numpy as np
import torch
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model.RDLE import RDLENet

from utils.dataread import CMELoader

TestDataset = CMELoader(img_dir='data', task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


# def main():
if not torch.cuda.is_available():
    print('no gpu device available')
    sys.exit(1)

model = RDLENet('./weights/RDLE_SAK.pt')
model = model.cuda()

model.eval()
with torch.no_grad():
    for _, (input, image_name) in enumerate(test_queue):
        input = Variable(input, volatile=True).cuda()
        image_name = image_name[0].split('\\')[-1].split('.')[0]
        i, r = model(input)
        u_name = '%s.png' % (image_name)
        print('processing {}'.format(u_name))
        u_path = 'results/' + u_name
        save_images(r, u_path)



# if __name__ == '__main__':
#     main()
