import imageio
import numpy as np
from tqdm import trange
from scipy.sparse import eye, kron, diags, vstack
from scipy.sparse.linalg import cg

from preprocess import preprocess

def FDmat(M, N):
    # Helper function to create a 2D finite difference matrix
    def create_2D_FDmat(size):
        e = np.ones(size)
        return diags([e, -e], [1, 0], shape=(size, size)).tocsc()

    # Vertical derivative
    S = diags(np.ones(N-1), offsets=1, shape=(N, N)).tocsc()
    T = create_2D_FDmat(M)
    Dy = kron(S, T)

    # Horizontal derivative
    S = create_2D_FDmat(N)
    T = diags(np.ones(M-1), offsets=1, shape=(M, M)).tocsc()
    Dx = kron(S, T)

    # Combine Dx and Dy
    combined = np.vstack([Dx.toarray(), Dy.toarray()])  # Convert to dense for concatenation
    return combined

def denoise_image(noise_image, alpha=0.1, max_iter=100, tol=1e-5, mode='cg', verbose=True):
    """
    Denoise an image using ADMM.
    """
    H, W, _ = noise_image.shape
    noise_image = noise_image.reshape(-1, noise_image.shape[-1])
    D = FDmat(H, W)
    obj_fn = lambda x: 0.5*np.linalg.norm(x - noise_image, 2)**2 + alpha * np.linalg.norm(D @ x, 1)

    if verbose:
        print("Starting ADMM...")
    
    x = noise_image.copy()
    z = D @ x
    u = np.zeros_like(z)
    c = 2.0
    I = eye(H*W, format='csc')
    DtD = D.T @ D
    eigenvalues, eigenvectors = np.linalg.eigh(DtD)

    for i in trange(max_iter):
        # Update x using a more efficient way
        b = noise_image + c * D.T @ (z - u)
        if mode == 'cg':
            A = I + c * DtD
            for i in range(x.shape[-1]):
                x_i, _ = cg(A, b[:, i], x0=x[:, i], tol=1e-5)
                x[:, i] = x_i

        elif mode == 'direct':
            lambda_inv = np.diag(1 / (1 + c * eigenvalues))
            A_inv = eigenvectors @ lambda_inv @ eigenvectors.T
            x = A_inv @ b

        # Update z
        Dx_plus_u = D @ x + u
        z = np.multiply(np.sign(Dx_plus_u), np.maximum(np.abs(Dx_plus_u) - alpha / c, 0), out=z)
        u = u + D @ x - z
        c = 1.1 * c
        
        # Check for convergence
        if np.linalg.norm(D @ x - z) < tol:
            if verbose:
                print(f'Converged at iteration {i}')
            break
        if verbose and i % 10 == 0:
            print(f'Iteration {i}, objective function value: {obj_fn(x)}')
    
    return x.reshape(H, W, -1)
    

def run(image_path, rgb2grey=False):
    """
    Run the denoising process.
    :param image_path: Path to the image to denoise.
    """
    ori_img, noise_image = preprocess(image_path, rgb2grey=rgb2grey, size=64)
    denoised_image = denoise_image(noise_image)

    # Compute the RMSE
    ori_rmse = np.sqrt(np.mean((ori_img - noise_image) ** 2))
    rmse = np.sqrt(np.mean((denoised_image - ori_img) ** 2))
    print(f'Ori RMSE: {ori_rmse:.4f}, Denoised RMSE: {rmse:.4f}')

    # Save the images
    denoised_image = (denoised_image + 1) / 2
    noise_image = (noise_image + 1) / 2
    ori_img = (ori_img + 1) / 2
    denoised_image = (denoised_image * 255).astype(np.uint8)
    noise_image = (noise_image * 255).astype(np.uint8)
    ori_img = (ori_img * 255).astype(np.uint8)

    denoised_image = denoised_image[..., 0] if rgb2grey else denoised_image
    noise_image = noise_image[..., 0] if rgb2grey else noise_image
    ori_img = ori_img[..., 0] if rgb2grey else ori_img

    imageio.imsave('assets/original_image.png', ori_img)
    imageio.imsave('assets/noise_image.png', noise_image)
    imageio.imsave('assets/denoised_image.png', denoised_image)

if __name__ == '__main__':
    run('assets/image.jpg')