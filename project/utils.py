import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_hdr_image(im_hdr):
    '''
    Maps the HDR intensities into a 0 to 1 range and then displays. 
    Three suggestions to try: 
      (1) Take log and then linearly map to 0 to 1 range (see display.py for example) 
    '''
    im_hdr = np.log(im_hdr + 0.05)
    im_hdr = (im_hdr - np.min(im_hdr)) / (np.max(im_hdr) - np.min(im_hdr))
    plt.axis('off')
    plt.title('HDR Image')
    plt.imsave('hdr_image.png', im_hdr)

def read_hdr_image(image_path: str) -> np.ndarray:
    '''
    Reads image from image path, and 
    return HDR floating point RGB image
    
    Args:
        image_path: path to hdr image

    Returns:
        RGB image of shape H x W x 3 in floating point format
    '''
    
    # read image and convert to RGB
    bgr_hdr_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    rgb_hdr_image = bgr_hdr_image[:, :, [2, 1, 0]]
    return rgb_hdr_image.astype(np.float32)


def write_hdr_image(hdr_image: np.ndarray, image_path: str):
    '''
    Write HDR image to given path.
    The path must end with '.hdr' extension
    Args:
        hdr_image: H x W x C float32 HDR image in BGR format.
        image_path: path to hdr image to save

    Returns:
        RGB image of shape H x W x 3 in floating point format
    '''
    assert(image_path.endswith('.hdr'))
    rgb_hdr_image = hdr_image[:, :, [2, 1, 0]]
    cv2.imwrite(image_path, rgb_hdr_image)