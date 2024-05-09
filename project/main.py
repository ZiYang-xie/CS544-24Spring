import argparse
from models import KANModel, MLPModel
from utils import read_config
from tasks import TASKS
from optimizers import build_optimizer

class TestBench():
    def __init__(self, args):
        self.config = read_config(args.config)
        
        self.task = TASKS[self.config['task']](**self.config['data'])
        self.dataset = self.task.create_dataset()

        self.kan = KANModel(**self.config['KAN'])
        self.mlp = MLPModel(**self.config['MLP'])


    def train(self):
        print("Training the model")
        # self.kan.train(self.dataset, self.config['optimizer']['name'])
        self.mlp.train(self.dataset,
                       build_optimizer(self.config['optimizer'], self.mlp.model.parameters()))
        

    def test(self):
        print("Testing the model")
        self.task.test(self.mlp, self.dataset)

    def run(self):
        self.train()
        self.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLP vs Kan Experiment')
    parser.add_argument('--config', type=str, help='path to config file')
    args = parser.parse_args()

    tb = TestBench(args)
    tb.run()