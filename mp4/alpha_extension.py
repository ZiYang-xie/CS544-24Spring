import gradio as gr
import cv2
import imageio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
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

def create_graph(img, mask, fg_prob, bg_prob, neighbors=4, lam=1):
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

    unary_cost = []
    binary_cost = []

    gamma = 2.0
    cnst = 1.0


    # Add edges between pixels and source/sink
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            idx = indices[y, x]
            capacity=fg_prob[y,x] # + lam*fg_dist[y,x]
            graph.add_edge('source', idx, capacity=capacity)
            unary_cost.append(capacity)
            capacity=bg_prob[y,x] # + lam*bg_dist[y,x]
            graph.add_edge(idx, 'sink', capacity=capacity)
            unary_cost.append(capacity)

            if x > 0:  # Left pixel
                left_idx = indices[y, x - 1]
                weight = (mask[y,x]!=mask[y,x-1])*gamma *np.exp(-1.0*(np.linalg.norm(img[y, x,:3] - img[y, x - 1,:3]) + 1e-6)) + cnst
                graph.add_edge(idx, left_idx, capacity=weight)
                binary_cost.append(weight)
                
            if y > 0:  # Upper pixel
                up_idx = indices[y - 1, x]
                weight = (mask[y,x]!=mask[y - 1, x])*gamma *np.exp(-1.0*(np.linalg.norm(img[y, x,:3] - img[y - 1, x,:3]) + 1e-6)) + cnst
                graph.add_edge(idx, up_idx, capacity=weight)
                binary_cost.append(weight)
            
            if neighbors==8 and x > 0 and y > 0:  # Upper-left pixel
                up_left_idx = indices[y - 1, x - 1]
                weight = (mask[y,x]!=mask[y - 1, x - 1])*gamma *np.exp(-1.0*(np.linalg.norm(img[y, x,:3] - img[y - 1, x - 1,:3]) + 1e-6)) / np.sqrt(2) + cnst
                graph.add_edge(idx, up_left_idx, capacity=weight)
                binary_cost.append(weight)
            
            if neighbors==8 and x < img.shape[1]-1 and y > 0:  # Upper-right pixel
                up_right_idx = indices[y - 1, x + 1]
                weight = (mask[y,x]!=mask[y - 1, x + 1])*gamma *np.exp(-1.0*(np.linalg.norm(img[y, x,:3] - img[y - 1, x + 1,:3]) + 1e-6)) / np.sqrt(2) + cnst
                graph.add_edge(idx, up_right_idx, capacity=weight)
                binary_cost.append(weight)
    logger.info(f"Unary Mean: {np.mean(unary_cost)}")
    logger.info(f"Unary Std: {np.std(unary_cost)}")
    logger.info(f"Binary Mean: {np.mean(binary_cost)}")
    logger.info(f"Binary Std: {np.std(binary_cost)}")
    return graph

def segment_image(image, masks, round=5):
    img = image
    tol = 1e-7
    arc_cut_value = -1e-9

    bg_mask = np.all(masks == 0, axis=0)
    masks = np.concatenate([masks, bg_mask[np.newaxis, ...]], axis=0)
    masks = masks.astype(bool)
    fg_gmm = GaussianMixture(n_components=2, random_state=0)
    bg_gmm = GaussianMixture(n_components=2, random_state=0)

    num_pixels_hist = [[np.count_nonzero(masks[i]) for i in range(masks.shape[0])]]

    for iter in range(masks.shape[0] * round):
        alpha = iter % masks.shape[0]
        alpha_mask = masks[alpha].copy()
        logger.info('Round: {}'.format(alpha))

        # pdb.set_trace()


        fg_data = image[alpha_mask] / 255.
        bg_data = image[~alpha_mask] / 255.
        fg_gmm.fit(fg_data)
        bg_gmm.fit(bg_data)

        # Compute the probability of each pixel belonging to the foreground and background
        data_eval = img.reshape(-1, 3) / 255.
        fg_prob, bg_prob = fg_gmm.score_samples(data_eval), bg_gmm.score_samples(data_eval)
        
        fg_prob = fg_prob.reshape(img.shape[:2])
        bg_prob = bg_prob.reshape(img.shape[:2])
        fg_prob = (fg_prob - np.min(fg_prob)) / (np.max(fg_prob) - np.min(fg_prob))
        bg_prob = (bg_prob - np.min(bg_prob)) / (np.max(bg_prob) - np.min(bg_prob))
        # fg_prob *= 2.0 / masks.shape[0]
        # bg_prob *= 2.0 * (masks.shape[0] - 1) / masks.shape[0]
        fg_prob, bg_prob = np.exp(fg_prob), np.exp(bg_prob)

        # Create a graph for the min-cut segmentation
        graph = create_graph(img, alpha_mask, fg_prob, bg_prob)
        cut_value, partition = nx.minimum_cut(graph, 'source', 'sink')
        reachable, non_reachable = partition
        logger.info(f"Cut value: {cut_value}")

        # Update the mask
        alpha_mask = np.zeros_like(alpha_mask)
        for segment in reachable:
            if segment != 'source':
                y, x = np.divmod(segment, img.shape[1])
                alpha_mask[y, x] = 1
        alpha_mask = alpha_mask.astype(bool)

        # Update masks
        for layer_idx in range(masks.shape[0]):
            new_mask = None
            if layer_idx == alpha:
                new_mask = alpha_mask | masks[layer_idx]  # Entry-wise operation
            else:
                new_mask = masks[layer_idx] & (~alpha_mask)  # Entry-wise operation
                # 1 & (0 | 1) = 1
                # 1 & (1 | 1) = 1
                # 0 & (0 | 0) = 0
                # 0 & (1 | 0) = 0
            masks[layer_idx] = new_mask.copy()
        for i in range(masks.shape[0]):
            print(np.count_nonzero(masks[i]))
        print('sum: ', np.sum(masks))
        num_pixels_hist.append([np.count_nonzero(masks[i]) for i in range(masks.shape[0])])
        if abs(cut_value - arc_cut_value) < tol:
            break
        arc_cut_value = cut_value



    logger.info('Segmentation done.')
    # plot num_pixels_hist
    plt.figure()
    num_pixels_hist = np.array(num_pixels_hist)
    for i in range(masks.shape[0]):
        plt.plot(num_pixels_hist[:, i], label=f'Layer {i}')
    plt.legend()
    plt.savefig('num_pixels_hist.png')
    return masks

def process(input, max_size=256, save_input_mask=True):
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
    else:
        image = ori_image

    masks = masks.astype(np.uint8)
    if save_input_mask:
        vis_mask = np.any(masks, axis=0).astype(np.uint8)
        vis_mask = cv2.resize(vis_mask, ori_image.shape[:2][::-1]) 
        vis_image = input['background'].copy()
        vis_image[vis_mask>0] = np.concatenate([vis_image[vis_mask>0][:,:3], 
                                            128*np.ones((vis_image[vis_mask>0].shape[0], 1), dtype=np.uint8)], axis=1)
        imageio.imwrite('input_mask.png', vis_image)
    
    refined_masks = segment_image(image, masks)

    segmented_image = input['background'].copy()
    mask = refined_masks[-1]
    mask = cv2.resize(mask.astype(np.uint8), ori_image.shape[:2][::-1]) == 1
    segmented_image[mask] = np.concatenate([segmented_image[mask][:,:3], 
                                            32*np.ones((segmented_image[mask].shape[0], 1), dtype=np.uint8)], axis=1)
    
    obj_masks = refined_masks[:-1]
    cmap = plt.cm.get_cmap('viridis', len(obj_masks))
    for i, mask in enumerate(obj_masks):
        overlay = Image.new("RGBA", ori_image.shape[:2][::-1], (int(255*cmap(i)[0]), int(255*cmap(i)[1]), int(255*cmap(i)[2]), 64))
        overlay = cv2.resize(np.array(overlay), ori_image.shape[:2][::-1])
        mask = cv2.resize(mask.astype(np.uint8), ori_image.shape[:2][::-1]) == 1
        overlay[~mask] = 0
        cv2.imwrite(f'overlay_{i}.png', overlay)
        segmented_image = Image.alpha_composite(Image.fromarray(segmented_image), Image.fromarray(overlay))
        segmented_image = np.array(segmented_image)

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
