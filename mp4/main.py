import gradio as gr
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_graph(img, mask, bgd_label, fgd_label, lam=50):
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
                graph.add_edge('source', idx, capacity=lam)
            elif mask[y, x] == bgd_label:
                graph.add_edge(idx, 'sink', capacity=lam)

            if x > 0:  # Left pixel
                left_idx = indices[y, x - 1]
                weight = lam / (np.linalg.norm(img[y, x] - img[y, x - 1]) + 1e-6)
                graph.add_edge(idx, left_idx, capacity=weight)
            if y > 0:  # Upper pixel
                up_idx = indices[y - 1, x]
                weight = lam / (np.linalg.norm(img[y, x] - img[y - 1, x]) + 1e-6)
                graph.add_edge(idx, up_idx, capacity=weight)

    return graph

def segment_image(image, mask):
    img = image
    mask = mask.astype(np.uint8)

    # Create a graph
    print("Creating graph...")
    graph = create_graph(img, mask, bgd_label=0, fgd_label=1)
    
    print('Computing min-cut...')
    cut_value, partition = nx.minimum_cut(graph, 'source', 'sink')
    reachable, non_reachable = partition
    print(f"Cut value: {cut_value}")

    segmented_img = np.zeros(img.shape, dtype=np.uint8)
    for segment in reachable:
        if segment != 'source':
            y, x = np.divmod(segment, img.shape[1])
            segmented_img[y, x] = img[y, x]

    return segmented_img

def process(input, max_size=512):
    #  Resize the longest edge to 512
    image = input['background']
    mask = input['layers'][0][:,:,-1]>0

    if max(image.shape[:2]) > max_size:
        scale = max_size / max(image.shape[:2])
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        mask = cv2.resize(mask.astype(np.uint8), (0, 0), fx=scale, fy=scale) > 0
    segmented_image = segment_image(image, mask)
    return segmented_image

if __name__ == "__main__":
    # Gradio interface
    iface = gr.Interface(
        fn=process,
        inputs=gr.ImageMask(image_mode='RGBA', sources=['upload']),
        outputs=gr.Image(),
        title="Interactive Image Segmentation",
        description="Draw on the image to mark the foreground in red and the background in blue, then see the segmentation results."
    )

    iface.launch()
