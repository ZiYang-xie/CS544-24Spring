import argparse
from models import KANModel, MLPModel
from utils import read_config
from tasks import TASKS
from optimizers import build_optimizer
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F

class TestBench():
    def __init__(self, args):
        self.config = read_config(args.config)
        
        self.task = TASKS[self.config['task']](self.config)

        self.kan = KANModel(**self.config['KAN'])
        self.mlp = MLPModel(**self.config['MLP'])
        self.dataset = {self.kan.name: self.task.create_dataset(self.kan.name),
                        self.mlp.name: self.task.create_dataset(self.mlp.name)}

    def train(self):
        print("Training the model")
        print(f"Before:KAN params: {sum(p.numel() for p in self.kan.model.parameters())}")
        print(f"MLP params: {sum(p.numel() for p in self.mlp.model.parameters())}")
        
        kan_losses = {}
        time_taken = {}
        import time
        for optimizer_name in ['SGD', 'Adam', 'LBFGS']:
            self.kan = KANModel(**self.config['KAN'])
            start = time.time()
            kan_loss = self.kan.train(self.dataset[self.kan.name], 
                            build_optimizer(self.config['optimizer'][optimizer_name], self.kan.model.parameters()),
                            iter=self.config['iterations'],
                            batch=self.config['batch_size'])
            end = time.time()
            kan_losses[optimizer_name] = kan_loss
            time_taken[optimizer_name] = end - start
        
        print(f"After:KAN params: {sum(p.numel() for p in self.kan.model.parameters())}")
        # plot the loss
        plt.figure()
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        # plt.ylim(0, min(2.4, max(kan_loss)))
        # plot both losses on the same graph with labels
        for optimizer_name in ['SGD', 'Adam', 'LBFGS']:
            plt.plot(kan_losses[optimizer_name], label=optimizer_name)
        plt.legend()
        # save the plot to a file
        plt.savefig(f"figs/{self.config['task']}/loss.png")
        plt.cla()
        # plot the time taken as a bar chart
        plt.bar(time_taken.keys(), time_taken.values())
        plt.title('Time for different optimizers')
        plt.xlabel('Optimizer')
        plt.ylabel('Time taken in seconds')
        plt.savefig(f"figs/{self.config['task']}/time.png")

    def test(self):
        print("Testing the model")
        self.task.test(self.mlp, self.dataset[self.mlp.name])
        self.task.test(self.kan, self.dataset[self.kan.name])

    def run(self):
        self.train()
        self.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLP vs Kan Experiment')
    parser.add_argument('--config', type=str, help='path to config file')
    args = parser.parse_args()

    tb = TestBench(args)
    tb.run()