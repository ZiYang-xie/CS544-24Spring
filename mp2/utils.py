import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os
def compose_images(image_list):
    shape = image_list[0].shape
    height = 480
    empty_images = np.zeros((height*len(image_list), *shape[1:]))
    for i, image in enumerate(image_list):
        empty_images[i*height:(i+1)*height] = image[0:480]
    cv2.imwrite("./scico_composed_img.jpg", empty_images)
    
if __name__ == "__main__":
    root_dir = "./scico_result"
    paths = os.listdir(root_dir)
    paths = [paths[i] for i in [3, 0, 4, 2, 1]]
    print(f"paths: {paths}")
    images = []
    for pth in paths:
        path = os.path.join(root_dir, pth)
        images.append(np.array(cv2.imread(path)))
    
    compose_images(images)