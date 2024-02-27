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

if __name__ == "__main__":
    start = time.time()
    image_path = 'assets/image.jpg'
    x_gt, y = preprocess(image_path, rgb2grey=False)
    print(f"x_gt: {x_gt.shape}, y: {y.shape}")
    N = x_gt.shape[0]
    y = (y + 1) / 2
    x_gt = (x_gt + 1) / 2

    
    # Solve ADMM
    λ = 0.5
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
            itstat_options={"display": True, "period": 10},
        )

        print(f"Solving on {device_info()}\n")
        x_tv[:, :, i] = solver.solve()
    print(f"Elapsed time: {time.time() - start:.2f} s")

    plt_args = {}
    fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(13, 12))
    plot.imview(x_gt, title="Ground truth", fig=fig, ax=ax[0], **plt_args)
    plot.imview(y, title="Noisy image", fig=fig, ax=ax[1], **plt_args)
    plot.imview(
        x_tv,
        title=f"ℓ1-TV denoising: {metric.mse(x_gt, x_tv)**0.5:.4f}",
        fig=fig,
        ax=ax[2],
        **plt_args,
    )

    fig.show()
    plot.savefig('scico_denoise.png')