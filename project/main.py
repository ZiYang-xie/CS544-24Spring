import argparse
from models import KANModel, MLPModel
from utils import read_config
from tasks import TASKS
from optimizers import build_optimizer
from matplotlib import pyplot as plt

class TestBench():
    def __init__(self, args):
        self.config = read_config(args.config)
        
        self.task = TASKS[self.config['task']](**self.config['data'])
        self.dataset = self.task.create_dataset()

        self.kan = KANModel(**self.config['KAN'])
        self.mlp = MLPModel(**self.config['MLP'])


    def train(self):
        print("Training the model")
        mlp_loss = self.mlp.train(self.dataset,
                       build_optimizer(self.config['optimizer'], self.mlp.model.parameters()))
        kan_loss = self.kan.train(self.dataset, 'LBFGS')        
        # plot the loss
        plt.figure()
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        # plot both losses on the same graph with labels
        plt.plot(kan_loss, label='KAN')
        plt.plot(mlp_loss, label='MLP')
        plt.legend()
        # save the plot to a file
        plt.savefig("loss.png")

    def test(self):
        print("Testing the model")
        self.task.test(self.mlp, self.dataset)

    def run(self):
        self.train()
        # self.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLP vs Kan Experiment')
    parser.add_argument('--config', type=str, help='path to config file')
    args = parser.parse_args()

    tb = TestBench(args)
    tb.run()