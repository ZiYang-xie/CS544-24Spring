import torch
import cv2
import random
import imageio
import numpy as np
import torch.nn.functional as F

class ImageFitting:
    def __init__(self, config, resize=96):
        self.path = config['data']['path']
        self.image = imageio.imread(self.path)[:, :, :3]
        H, W = self.image.shape[:2]
        self.image = cv2.resize(self.image, (int(resize*W/H), resize) if H > W else (resize, int(resize*H/W)))
        # save the image to a file
        imageio.imsave("input.png", self.image)
        self.image = self.image / 255.0
        self.config = config
        random.seed(42)

    def input_mapping(self, input, scale=10.0, mapping_size=16):
        B = np.random.normal(0, 1, (mapping_size, input.shape[1])) * scale
        x_proj = (2.*np.pi*input) @ B.T
        return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

    def create_dataset(self, model_name):
        x = np.linspace(0, 1, self.image.shape[1])
        y = np.linspace(0, 1, self.image.shape[0])
        X, Y = np.meshgrid(x, y)
        grid = np.stack([X, Y], axis=-1)
        input = grid.reshape(-1, 2)
        input = self.input_mapping(input, mapping_size=self.config[model_name]['width'][0]//2)
        output = self.image.reshape(-1, 3)

        input = torch.tensor(input, dtype=torch.float32)
        output = torch.tensor(output, dtype=torch.float32)

        # Shuffle the dataset
        idx = np.arange(input.shape[0])
        np.random.shuffle(idx)
        shuffled_input = input[idx]
        shuffled_output = output[idx]

        dataset = {
            'train_input': shuffled_input,
            'train_label': shuffled_output,
            'test_input': input,
            'test_label': output
        }
        # print the shapes of the dataset
        for key in dataset:
            print(key, dataset[key].shape)
        return dataset
    
    def plot(self, method, dataset):
        pred = method.model(dataset['test_input'])
        pred = pred.clamp(0, 1)
        pred = pred.detach().cpu().numpy().reshape(self.image.shape)
        pred = (pred * 255).astype(np.uint8)
        import os
        if not os.path.exists(f"figs/{self.config['task']}"):
            os.makedirs(f"figs/{self.config['task']}")
        imageio.imsave(f"figs/{self.config['task']}/{method}_output.png", pred)
    
    def test(self, method, dataset, plot=True):
        test_dict = method.test(dataset)
        pred = test_dict['pred']
        pred = pred.clamp(0, 1)
        pred = pred.detach().cpu().numpy().reshape(self.image.shape)
        pred = (pred * 255).astype(np.uint8)
        import os
        if not os.path.exists(f"figs/{self.config['task']}"):
            os.makedirs(f"figs/{self.config['task']}")
        imageio.imsave(f"figs/{self.config['task']}/{method.name}_output.png", pred)

        # self.plot(method, dataset)


    