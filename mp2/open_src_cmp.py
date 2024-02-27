import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from preprocess import preprocess

image_path = 'assets/image.jpg'
ori_img, noise_image = preprocess(image_path, rgb2grey=False)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5),
                    sharex=True, sharey=True)

# Save the images
noise_image = (noise_image + 1) / 2
ori_img = (ori_img + 1) / 2

plt.gray()

ax[0, 0].imshow(noise_image)
ax[0, 0].axis('off')
ax[0, 0].set_title('noise_image')
ax[0, 1].imshow(denoise_tv_chambolle(noise_image, weight=0.1, channel_axis=-1))
ax[0, 1].axis('off')
ax[0, 1].set_title('TV')
# ax[0, 2].imshow(denoise_bilateral(noise_image, sigma_color=0.05, sigma_spatial=15,
#                 channel_axis=-1))
# ax[0, 2].axis('off')
# ax[0, 2].set_title('Bilateral')
# ax[0, 3].imshow(denoise_wavelet(noise_image, channel_axis=-1, rescale_sigma=True))
# ax[0, 3].axis('off')
# ax[0, 3].set_title('Wavelet denoising')

ax[1, 1].imshow(denoise_tv_chambolle(noise_image, weight=0.2, channel_axis=-1))
ax[1, 1].axis('off')
ax[1, 1].set_title('(more) TV')
# ax[1, 2].imshow(denoise_bilateral(noise_image, sigma_color=0.1, sigma_spatial=15,
#                 channel_axis=-1))
# ax[1, 2].axis('off')
# ax[1, 2].set_title('(more) Bilateral')
# ax[1, 3].imshow(denoise_wavelet(noise_image, channel_axis=-1, convert2ycbcr=True,
#                                 rescale_sigma=True))
# ax[1, 3].axis('off')
# ax[1, 3].set_title('Wavelet denoising\nin YCbCr colorspace')
ax[1, 0].imshow(ori_img)
ax[1, 0].axis('off')
ax[1, 0].set_title('ori_img')

fig.tight_layout()
plt.show()
plt.savefig('opensrc.png')
