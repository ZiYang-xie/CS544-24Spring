from xdesign import SiemensStar, discrete_phantom
import numpy as np
import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import spnoise
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info
from scipy.ndimage import median_filter
from preprocess import preprocess
import time
plot.config_notebook_plotting()

def scico_optimize(image_path):
    start = time.time()
    x_gt, y = preprocess(image_path, rgb2grey=False)
    N = x_gt.shape[0]
    y = (y + 1) / 2
    x_gt = (x_gt + 1) / 2

    # Solve ADMM
    λ = 0.3
    x_tv = np.zeros(y.shape)
    for i in range(3):
        g_loss = loss.Loss(y=y[:,:,i], f=functional.L1Norm())
        g_tv = λ * functional.L1Norm()
        C = linop.FiniteDifference(input_shape=x_gt[:,:,i].shape, append=0)
        
        solver = ADMM(
            f=None,
            g_list=[g_loss, g_tv],
            C_list=[linop.Identity(input_shape=y[:,:,i].shape), C],
            rho_list=[2.0, 2.0],
            x0=y[:, :, i],
            maxiter=100,
            subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-7, "maxiter": 100}),
            itstat_options={"display": True, "period": 50},
        )

        print(f"Solving on {device_info()}\n")
        x_tv[:, :, i] = solver.solve()
    print(f"Elapsed time: {time.time() - start:.2f} s")

    plt_args = {}
    fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 5))
    plot.imview(x_gt, title="Original Image", fig=fig, ax=ax[0], **plt_args)
    plot.imview(y, title="Noisy image", fig=fig, ax=ax[1], **plt_args)
    plot.imview(
        x_tv,
        title=f"L1-TV Denoised Image",
        fig=fig,
        ax=ax[2],
        **plt_args,
    )
    ori_rmse = np.sqrt(np.mean((x_gt - y) ** 2))
    rmse = np.sqrt(np.mean((x_gt - x_tv) ** 2))
    fig.suptitle(f'Ori RMSE: {ori_rmse:.4f}, Denoised RMSE: {rmse:.4f}')
    fig.show()
    plot.savefig(f"scico_result/{image_path.split('/')[-1]}")

if __name__ == "__main__":
    import os
    images = os.listdir('assets')
    for image in images:
        print(f"Optimizing {image}")
        scico_optimize(f'assets/{image}')