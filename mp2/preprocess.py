import numpy as np
import cv2
import sys
import argparse

def main(args):
    name = args.filename.split('/')[-1].split('.')[0]
    img = cv2.imread(args.filename)
    img = cv2.resize(img, (args.size, args.size))
    if args.RGB2Grey:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(name + '_grey.png', img)
    
    img = np.array(img, dtype=np.float64)
    noise = np.random.normal(0, 1, img.shape)
    noise -= np.mean(noise)
    noise /= np.std(noise)
    noise *= args.noiseScale
    img += noise
    img = np.round(img)
    img = np.mod(img, 256)
    cv2.imwrite(name + '_noisy.png', img.astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='input image filename')
    parser.add_argument('--RGB2Grey', action='store_true', help='convert RGB to Grey')
    parser.add_argument('--size', type=int, default=64, help='resize image to size x size')
    parser.add_argument('--noiseScale', type=float, default=5, help='noise scale')
    parser.add_argument('--ourdir', type=str, default='./assets', help='output directory')
    args = parser.parse_args()

    print(f'filename: {args.filename}, RGB2Grey: {args.RGB2Grey}, size: {args.size}, noiseScale: {args.noiseScale}')
    main(args)
    

