import os
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# 28 x 28 images
class MNIST():
    def __init__(self, config):
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

    def create_dataset(self):
        # Flatten training and test datasets
        train_data_flat, train_targets = self.flatten_images(self.mnist_train)
        test_data_flat, test_targets = self.flatten_images(self.mnist_test)
        train_targets = train_targets.reshape(-1, 1)
        test_targets = test_targets.reshape(-1, 1)
        # convert to one hot encoding
        train_targets = torch.zeros(train_targets.size(0), 10).scatter_(1, train_targets, 1)
        test_targets = torch.zeros(test_targets.size(0), 10).scatter_(1, test_targets, 1)
        
        # Split training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(train_data_flat, train_targets, test_size=0.2, random_state=42)
        
        dataset = {
            'train_input': X_train,
            'train_label': y_train,
            'val_input': X_val,
            'val_label': y_val,
            'test_input': test_data_flat,
            'test_label': test_targets
        }
        for key in dataset:
            print(key, dataset[key].shape)
        return dataset

    def test(self, model, X, Y):
        probs = model.test(X)
        probs = probs['pred']
        probs = probs.detach().cpu().numpy()
        Y_h = probs.argmax(axis=1)
        # get precision and recall for each class
        precision = []
        recall = []
        for i in range(10):
            TP = ((Y_h == i) & (Y == i)).sum()
            FP = ((Y_h == i) & (Y != i)).sum()
            FN = ((Y_h != i) & (Y == i)).sum()
            precision.append(TP / (TP + FP))
            recall.append(TP / (TP + FN))
        # plot the precision and recall, save the plot to a file
        plt.figure()
        plt.plot(precision, label='precision')
        plt.plot(recall, label='recall')
        plt.legend()
        plt.savefig(f"{model.name}_output.png")

