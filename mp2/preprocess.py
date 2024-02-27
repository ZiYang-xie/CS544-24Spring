import numpy as np
import imageio
import cv2
import argparse

def preprocess(filename,
                size=64,
                noise_scale=0.1,
                rgb2grey=True, 
                save=False):
    img = imageio.imread(filename)
    ori_img = cv2.resize(img, (size, size))
    if rgb2grey:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        ori_img = np.expand_dims(ori_img, axis=-1)
    
    ori_img = ori_img / 255.0
    ori_img = ori_img * 2 - 1
    img = np.array(ori_img, dtype=np.float64)
    noise = np.random.normal(0, 1, img.shape)
    noise -= np.mean(noise)
    noise /= np.std(noise)
    noise *= noise_scale
    noised_img = img + noise
    noised_img = np.clip(noised_img, -1, 1)

    return ori_img, noised_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='input image filename')
    parser.add_argument('--RGB2Grey', action='store_true', help='convert RGB to Grey')
    parser.add_argument('--size', type=int, default=64, help='resize image to size x size')
    parser.add_argument('--noiseScale', type=float, default=5, help='noise scale')
    parser.add_argument('--ourdir', type=str, default='./assets', help='output directory')
    args = parser.parse_args()

    print(f'filename: {args.filename}, RGB2Grey: {args.RGB2Grey}, size: {args.size}, noiseScale: {args.noiseScale}')
    preprocess(args)
    

