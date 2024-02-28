import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os
def compose_images(image_list):
    shape = image_list[0].shape
    empty_images = np.zeros((shape[0]*len(image_list), *shape[1:]))
    for i, image in enumerate(image_list):
        empty_images[i*shape[0]:(i+1)*shape[0]] = image
    cv2.imwrite("./composed_img.jpg", empty_images)
    
if __name__ == "__main__":
    paths = os.listdir("./results")
    print(f"paths: {paths}")
    images = []
    for pth in paths:
        path = os.path.join("./results", pth)
        images.append(np.array(cv2.imread(path)))
    
    compose_images(images)