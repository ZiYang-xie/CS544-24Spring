import os
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import yaml
import numpy as np

# 28 x 28 images
class ImageClassification:
    def __init__(self, config):
        self.config = config
        data_path = config['data']['path']
        mnist_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        mnist_test = datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
        self.path = data_path
        self.mnist_train = mnist_train
        self.mnist_test = mnist_test

    def flatten_images(self, dataset):
        # Flatten each image in the dataset to a 1D tensor
        data_flat = []
        targets = []
        for img, target in dataset:
            img_flat = img.view(-1)
            data_flat.append(img_flat)
            targets.append(target)
        return torch.stack(data_flat), torch.tensor(targets)

    def create_dataset(self, model_name):
        # Flatten training and test datasets
        train_data_flat, train_targets = self.flatten_images(self.mnist_train)
        test_data_flat, test_targets = self.flatten_images(self.mnist_test)
        
        # # Split training data into training and validation sets
        # X_train, X_val, y_train, y_val = train_test_split(train_data_flat, train_targets, test_size=0.2, random_state=42)
        
        dataset = {
            'train_input': train_data_flat,
            'train_label': train_targets,
            # 'val_input': X_val,
            # 'val_label': y_val,
            'test_input': test_data_flat,
            'test_label': test_targets
        }
        for key in dataset:
            print(key, dataset[key].shape)
        return dataset

    def test(self, method, dataset):
        probs = method.test(dataset)
        probs = probs['pred']
        probs = probs.detach().cpu().numpy()
        Y_h = probs.argmax(axis=1)
        Y = dataset['test_label'].detach().cpu().numpy().astype(np.int32)
        # get precision and recall for each class
        precision = []
        recall = []
        for i in range(10):
            TP = ((Y_h == i) & (Y == i)).sum()
            FP = ((Y_h == i) & (Y != i)).sum()
            FN = ((Y_h != i) & (Y == i)).sum()
            precision.append(TP / (TP + FP))
            recall.append(TP / (TP + FN))
        accuracy = (Y_h == Y).mean()
        print(f"test: {method.name} overall accuracy: {accuracy}")
        # plot the precision and recall, save the plot to a file
        plt.figure()
        plt.plot(precision, label='precision')
        plt.plot(recall, label='recall')
        plt.legend()
        if not os.path.exists(f"figs/{self.config['task']}"):
            os.makedirs(f"figs/{self.config['task']}")
        plt.savefig(f"figs/{self.config['task']}/{method.name}_metrics.png")

if __name__ == "__main__":
    with open('configs/image_classification.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    mnist = ImageClassification(config)
    dataset = mnist.create_dataset()
    mnist.test(None, dataset['test_input'], dataset['test_label'])