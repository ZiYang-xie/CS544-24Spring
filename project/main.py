import argparse
import imageio
import os
from utils import read_hdr_image, write_hdr_image, display_hdr_image


class LDR2HDR:
    def __init__(self, args):
        self.data_path = args.data_path
        self.index = args.index
        self.hdr_image = self.read_hdr_image()
        self.ldr_images = self.read_ldr_images()

    def read_hdr_image(self):
        im_hdr = read_hdr_image(self.data_path + '/HDR/' + f'HDR_{int(self.index):03d}' + '.hdr')
        # display_hdr_image(im_hdr)
        return im_hdr

    def read_ldr_images(self):
        images = []
        for path in os.listdir(self.data_path):
            if 'LDR' not in path:
                continue
            ldr_path = self.data_path + f'/{path}/LDR_{int(self.index):03d}.jpg'
            ldr_image = imageio.imread(ldr_path)    
            images.append(ldr_image)
        return images



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display HDR image')
    parser.add_argument('--index', type=int, help='Index of the image to process')
    parser.add_argument('--data_path', type=str, default='./LDR-HDR-pair_Dataset')
    args = parser.parse_args()

    ldr_converter = LDR2HDR(args)





