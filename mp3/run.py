import os
import argparse 
import wandb

from models import MODEL_ZOO
from utils import generate_problem
import matplotlib.pyplot as plt
import numpy as np

def run(args):
    c, A, b = generate_problem(num_eq=args.num_eqs, v_num=args.num_vars)

    result_dict = {}
    for model_name in MODEL_ZOO.keys():
        print(f"Running {model_name}...")
        model = MODEL_ZOO[model_name](c, A, b, use_wandb=args.wandb)
        result = model.solve(tol=args.tol, max_iter=args.max_iter)
        print(f"Optimal value: {result['value']}")
        print(f"Optimal x: {result['x']}")
        if 'duality_measure' in result:
            print(f"Duality measure: {result['duality_measure']}")
        print("\n\n")
        result_dict[model_name] = result

    # Export the results in json
    if not os.path.exists('./output'):
        os.makedirs('./output')

    # Save the results
    with open(f'./output/res_v{args.num_vars}_e{args.num_eqs}.txt', 'w') as f:
        for model in result_dict.keys():
            f.write(f'{model}:\n')
            f.write(f'Optimal value: {result_dict[model]["value"]}\n')
            if 'duality_measure' in result_dict[model]:
                f.write(f'Duality measure: {result_dict[model]["duality_measure"]}\n')
            f.write('\n\n')

    # Visualize the results, compare them on the same plot
    if args.vis:
        for model in result_dict.keys():
            plt.plot(result_dict[model]['value_history'], label=model)
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Value Convergence of Different Models')
        plt.savefig(f'./output/res_v{args.num_vars}_e{args.num_eqs}.png')
        plt.close()
        plt.cla()

        for model in result_dict.keys():
            if 'rp_norm_history' in result_dict[model]:
                plt.plot(result_dict[model]['rp_norm_history'], label=model)
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Primal Residual Norm')
        plt.title('Primal Residual Norm Convergence of Different Models')
        plt.savefig(f'./output/res_v{args.num_vars}_e{args.num_eqs}_primal.png')
        plt.close()
        plt.cla()

        for model in result_dict.keys():
            if 'duality_measure_history' in result_dict[model]:
                plt.plot(result_dict[model]['duality_measure_history'], label=model)
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Duality Measure')
        plt.title('Duality Measure Convergence of Different Models')
        plt.savefig(f'./output/res_v{args.num_vars}_e{args.num_eqs}_duality.png')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS544-24Spring/mp3')
    parser.add_argument('--vis', action='store_true', help='Visualize the optimization process')
    parser.add_argument('--num_vars', type=int, default=30, help='Number of variables')
    parser.add_argument('--num_eqs', type=int, default=10, help='Number of equations')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance for convergence')
    parser.add_argument('--max_iter', type=int, default=500, help='Maximum number of iterations')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    args = parser.parse_args()

    run(args)

