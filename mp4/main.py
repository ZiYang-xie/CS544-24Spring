import gradio as gr
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("rich")

def create_graph(img, mask, fg_prob, bg_prob, fgd_label=1, bgd_label=0):
    """
    Create a graph for the min-cut segmentation.
    """
    indices = np.arange(img.size // img.shape[2]).reshape(img.shape[:2])
    graph = nx.Graph()

    # Add edges between pixels and source/sink
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            idx = indices[y, x]
            if mask[y, x] == fgd_label:
                graph.add_edge('source', idx, capacity=fg_prob[y,x])
            elif mask[y, x] == bgd_label:
                graph.add_edge(idx, 'sink', capacity=bg_prob[y,x])

            if x > 0:  # Left pixel
                left_idx = indices[y, x - 1]
                weight = (mask[y,x]!=mask[y,x-1])*np.exp(-1.0*(np.linalg.norm(img[y, x,:3] - img[y, x - 1,:3]) + 1e-6))
                graph.add_edge(idx, left_idx, capacity=weight)
            if y > 0:  # Upper pixel
                up_idx = indices[y - 1, x]
                weight = (mask[y,x]!=mask[y - 1, x])*np.exp(-1.0*(np.linalg.norm(img[y, x,:3] - img[y - 1, x,:3]) + 1e-6))
                graph.add_edge(idx, up_idx, capacity=weight)

    return graph

def segment_image(image, gmm_model, round=5):
    img = image

    tol = 1e-7
    arc_cut_value = -999999
    # Create a graph
    for _ in range(round):
        logger.info('Round: {}'.format(_))

        # Compute the probability of each pixel belonging to the foreground and background
        fg_prob, bg_prob =  gmm_model['fg'].score_samples(img.reshape(-1, 3)), gmm_model['bg'].score_samples(img.reshape(-1, 3))
        fg_prob, bg_prob = np.exp(fg_prob), np.exp(bg_prob)
        mask = (fg_prob > bg_prob)
        mask = mask.reshape(img.shape[:2]).astype(np.uint8)
        fg_prob = fg_prob.reshape(img.shape[:2])
        bg_prob = bg_prob.reshape(img.shape[:2])

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
        mask = mask.astype(bool)
        fg_data = img[mask]
        bg_data = img[~mask]
        gmm_model['fg'].fit(fg_data)
        gmm_model['bg'].fit(bg_data)

        if abs(cut_value - arc_cut_value) < tol:
            break
        arc_cut_value = cut_value

    logger.info('Segmentation done.')
    return mask

def process(input, max_size=256):
    ori_image = input['background'][:,:,:3]
    mask = input['layers'][0][:,:,-1]>0

    fg_gmm = GaussianMixture(n_components=2, random_state=0)
    bg_gmm = GaussianMixture(n_components=2, random_state=0)
    fg_data = ori_image[mask]
    bg_data = ori_image[~mask]
    fg_gmm.fit(fg_data)
    bg_gmm.fit(bg_data)

    gmm_model = {
        'fg': fg_gmm,
        'bg': bg_gmm
    }

    # Resize the image if it is too large
    if max(ori_image.shape[:2]) > max_size:
        scale = max_size / max(ori_image.shape[:2])
        image = cv2.resize(ori_image, (0, 0), fx=scale, fy=scale)
        mask = cv2.resize(mask.astype(np.uint8), (0, 0), fx=scale, fy=scale) > 0

    refined_mask = segment_image(image, gmm_model=gmm_model)
    mask = cv2.resize(refined_mask.astype(np.uint8), ori_image.shape[:2][::-1]) == 0
    segmented_image = input['background'].copy()
    segmented_image[mask] = np.concatenate([segmented_image[mask][:,:3], 
                                            32*np.ones((segmented_image[mask].shape[0], 1), dtype=np.uint8)], axis=1)
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
