import torch
import cv2
import random
import imageio
import numpy as np
import torch.nn.functional as F
import yaml
class ContinueFitting:
    def __init__(self, config, resize=96):
        self.path1 = config['data']['path1']
        self.image1 = imageio.imread(self.path1)[:, :, :3]
        H, W = self.image1.shape[:2]
        self.image1 = cv2.resize(self.image1, (int(resize*W/H), resize) if H > W else (resize, int(resize*H/W)))
        # save the image to a file
        # imageio.imsave("input.png", self.image1)
        self.image1 = self.image1 / 255.0
        self.H = self.image1.shape[0]
        self.W = self.image1.shape[1]
        
        self.path2 = config['data']['path2']
        self.image2 = imageio.imread(self.path2)[:, :, :3]
        self.image2 = cv2.resize(self.image2, (self.W, self.H))
        # save the image to a file
        # imageio.imsave("input.png", self.image2)
        self.image2 = self.image2 / 255.0
        self.images = [self.image1, self.image2]
        
        self.config = config
        random.seed(42)

    def input_mapping(self, input, mapping_embedding, scale=10.0):
        x_proj = (2.*np.pi*input) @ mapping_embedding.T *scale
        return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

    def create_dataset(self, model_name):
        mapping_embedding = np.random.normal(0, 1, (self.config[model_name]['width'][0]//2, 2))
        dataset_list = []
        for i in range(len(self.images)):
            x = np.linspace(i, i+1, self.W)
            y = np.linspace(0, 1, self.H)
            X, Y = np.meshgrid(x, y)
            grid = np.stack([X, Y], axis=-1)
            input = grid.reshape(-1, 2)
            input = self.input_mapping(input, mapping_embedding)
            # input embedding
            # scale = 10.0
            # x_proj = (2.*np.pi*input) @ mapping_embedding.T *scale
            # input = np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1) + 2*i - 1
            
            output = self.images[i].reshape(-1, 3)
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
            dataset_list.append(dataset)
            # print the shapes of the dataset
            for key in dataset:
                print(key, dataset[key].shape)
        # combine the two datasets
        combine_dataset = {}
        combine_dataset['train_input'] = torch.cat([dataset_list[0]['train_input'], dataset_list[1]['train_input']], axis=0)
        combine_dataset['train_label'] = torch.cat([dataset_list[0]['train_label'], dataset_list[1]['train_label']], axis=0)
        combine_dataset['test_input'] = torch.cat([dataset_list[0]['test_input'], dataset_list[1]['test_input']], axis=0)
        combine_dataset['test_label'] = torch.cat([dataset_list[0]['test_label'], dataset_list[1]['test_label']], axis=0)
        # dataset_list = [dataset_list[0], dataset_list[0]]
        return dataset_list
    
    def plot(self, method, dataset):
        pred = method.model(dataset['test_input'])
        pred = pred.clamp(0, 1)
        pred = pred.detach().cpu().numpy().reshape(self.image1.shape)
        pred = (pred * 255).astype(np.uint8)
        imageio.imsave(f"{method}_output.png", pred)
    
    def test(self, method, dataset_list, plot=True):
        print(f"Continue Fitting testing...")
        pred_list = []
        for phase, dataset in enumerate(dataset_list):
            print(f"phase:{phase}")
            test_dict = method.test(dataset)
            pred = test_dict['pred']
            pred = pred.clamp(0, 1)
            pred = pred.detach().cpu().numpy().reshape(self.image1.shape)
            pred = (pred * 255).astype(np.uint8)
            pred_list.append(pred)
        output_img = np.ones((self.H, self.W*2+10, 3), dtype=np.uint8)*255
        for i, pred in enumerate(pred_list):
            output_img[:, i*self.W+i*10:(i+1)*self.W+i*10] = pred
        imageio.imsave(f"figs/continue_fitting/{method.name}_output.png", output_img)
        # self.plot(method, dataset)

if __name__ == "__main__":
    with open('configs/continue_fitting.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    mnist = ContinueFitting(config)
    dataset = mnist.create_dataset('MLP')
    mnist.test(None, dataset['test_input'], dataset['test_label'])
    