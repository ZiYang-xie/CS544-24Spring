import gradio as gr
import cv2
import imageio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.ndimage import distance_transform_edt

import logging
from rich.logging import RichHandler
import pdb

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("rich")

def find_nearest_value_dist(matrix, value):
    matrix = np.array(matrix)
    target_positions = (matrix == value)
    distances = distance_transform_edt(~target_positions)
    distances = np.round(distances).astype(int)
    return distances

def create_graph(img, mask, fg_prob, bg_prob, lam=1):
    """
    Create a graph for the min-cut segmentation.
    """
    indices = np.arange(img.size // img.shape[2]).reshape(img.shape[:2])
    graph = nx.Graph()

    img = img / 255.
    fg_dist = find_nearest_value_dist(mask, 1)
    bg_dist = find_nearest_value_dist(mask, 0)

    H, W = img.shape[:2]
    fg_dist = 1 - fg_dist / max(H, W)
    bg_dist = 1 - bg_dist / max(H, W)

    # Add edges between pixels and source/sink
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            idx = indices[y, x]
            graph.add_edge('source', idx, capacity=fg_prob[y,x] + lam*fg_dist[y,x])
            graph.add_edge(idx, 'sink', capacity=bg_prob[y,x] + lam*bg_dist[y,x])

            if x > 0:  # Left pixel
                left_idx = indices[y, x - 1]
                weight = (mask[y,x]!=mask[y,x-1])*np.exp(-1.0*(np.linalg.norm(img[y, x,:3] - img[y, x - 1,:3]) + 1e-6))
                graph.add_edge(idx, left_idx, capacity=weight)

            if y > 0:  # Upper pixel
                up_idx = indices[y - 1, x]
                weight = (mask[y,x]!=mask[y - 1, x])*np.exp(-1.0*(np.linalg.norm(img[y, x,:3] - img[y - 1, x,:3]) + 1e-6))
                graph.add_edge(idx, up_idx, capacity=weight)
    return graph

def segment_image(image, masks, round=1):
    img = image
    tol = 1e-7
    arc_cut_value = -1e-9

    bg_mask = np.all(masks == 0, axis=0)
    masks = np.concatenate([masks, bg_mask[np.newaxis, ...]], axis=0)
    masks = masks.astype(bool)
    fg_gmm = GaussianMixture(n_components=2, random_state=0)
    bg_gmm = GaussianMixture(n_components=2, random_state=0)

    for iter in range(masks.shape[0] * round):
        sc_layer_idx = iter % masks.shape[0]
        mask = masks[sc_layer_idx].copy()
        logger.info('Round: {}'.format(sc_layer_idx))

        # pdb.set_trace()


        fg_data = image[mask] / 255.
        bg_data = image[~mask] / 255.
        fg_gmm.fit(fg_data)
        bg_gmm.fit(bg_data)

        # Compute the probability of each pixel belonging to the foreground and background
        data_eval = img.reshape(-1, 3) / 255.
        fg_prob, bg_prob = fg_gmm.score_samples(data_eval), bg_gmm.score_samples(data_eval)
        
        fg_prob = fg_prob.reshape(img.shape[:2])
        bg_prob = bg_prob.reshape(img.shape[:2])
        fg_prob = (fg_prob - np.min(fg_prob)) / (np.max(fg_prob) - np.min(fg_prob))
        bg_prob = (bg_prob - np.min(bg_prob)) / (np.max(bg_prob) - np.min(bg_prob))
        fg_prob, bg_prob = np.exp(fg_prob), np.exp(bg_prob)

        # Create a graph for the min-cut segmentation
        graph = create_graph(img, mask, fg_prob, bg_prob)
        cut_value, partition = nx.minimum_cut(graph, 'source', 'sink')
        reachable, non_reachable = partition
        logger.info(f"Cut value: {cut_value}")

        # Update the mask
        mask = np.zeros_like(mask)
        for segment in reachable:
            if segment != 'source':
                y, x = np.divmod(segment, img.shape[1])
                mask[y, x] = 1

        # Update the GMM model
        # mask = mask.astype(bool)
        # fg_data = img[mask] / 255.
        # bg_data = img[~mask] / 255.
        # fg_gmm.fit(fg_data)
        # bg_gmm.fit(bg_data)

        # Update masks
        for layer_idx in range(masks.shape[0]):
            if layer_idx == sc_layer_idx:
                masks[layer_idx] = mask  # Just copy the mask
            else:
                masks[layer_idx] = masks[layer_idx] & (~mask | masks[layer_idx])  # Entry-wise operation

        if abs(cut_value - arc_cut_value) < tol:
            break
        arc_cut_value = cut_value



    logger.info('Segmentation done.')
    return masks

def process(input, max_size=256):
    ori_image = input['background'][:,:,:3]
    masks = np.array(input['layers'])[...,-1]>0

    # Resize the image if it is too large
    if max(ori_image.shape[:2]) > max_size:
        scale = max_size / max(ori_image.shape[:2])
        image = cv2.resize(ori_image, (0, 0), fx=scale, fy=scale)
        resized_masks = []
        for mask in masks:
            resized_masks.append(cv2.resize(mask.astype(np.uint8), (0, 0), fx=scale, fy=scale) > 0)
        masks = np.array(resized_masks)

    masks = masks.astype(np.uint8)
    refined_masks = segment_image(image, masks)
    import pdb; pdb.set_trace()
    mask = refined_masks[-1]
    mask = cv2.resize(mask.astype(np.uint8), ori_image.shape[:2][::-1]) == 0

    segmented_image = input['background'].copy()
    segmented_image[mask] = np.concatenate([segmented_image[mask][:,:3], 
                                            32*np.ones((segmented_image[mask].shape[0], 1), dtype=np.uint8)], axis=1)

    # Define colors for each mask
    # num_masks = masks.shape[0]  # Assuming masks is a numpy array with shape (num_masks, height, width)
    # colors = np.random.randint(0, 256, size=(num_masks, 3), dtype=np.uint8)  # Generate random colors for each mask

    # Iterate over each mask and assign a different color
    # for i in range(num_masks):
    #     mask = masks[i]
    #     mask = mask.astype(bool)
    #     color = colors[i]
    #     segmented_image[mask] = color
    # segmented_image = np.concatenate([segmented_image, 32*np.ones((segmented_image.shape[0], segmented_image.shape[1], 1), dtype=np.uint8)], axis=2)    
        
    return segmented_image

if __name__ == "__main__":
    # Gradio interface
    iface = gr.Interface(
        fn=process,
        inputs=gr.ImageMask(image_mode='RGBA', sources=['upload']),
        outputs=gr.Image(image_mode='RGBA'),
        title="Interactive Image Segmentation",
        description="Draw on the image to mark the foreground",
        allow_flagging=False
    )

    iface.launch()
