import numpy as np
import os
from PIL import Image
from util import init_seg_needed, output, save_init_seg
from rag_builder import build_rag
from skimage.segmentation import slic
from skimage.future import graph

def main(imtitle, imtype, weight_mode = 'similarity', compactness = 30, n_segments = 130, thresh=1e-3, beta="1", slic_output = False, seg_output = True
, slic_output_mode='show', seg_output_mode='show'):
    '''
    Performs a clustering of input image into superpixel, then performs
    an n-cut segmentation on the region adjacency graph corresponding
    to these. Data for the superpixels is saved, so that the clustering does not
    need to be performed unnecessarily.

    Parameters
    ----------

    imtitle : string
        The title of the image file without specification for format, i.e. the text before .jpg, .png etc.

    imtype : string
        The format of the image, e.g. jpg, png etc.

    weight_mode : string
        The desired method for graph construction. Possible choices are 'similarity' and 'laplacian'.

    compactness : int
        Balances color and space proximity of SLIC clustering.
        More info: https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/slic_superpixels.py

    n_segments : int
        Number of desired superpixels from the SLIC clustering.
        More info: https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/slic_superpixels.py

    thresh : int
        Threshold for when to further dived subgraphs in the Ncut segmentation.
        More info: https://github.com/scikit-image/scikit-image/blob/master/skimage/future/graph/graph_cut.py

    beta : string
        If weight_mode is selected to 'laplacian' specifies the beta used to generate the laplacian in question.
        If a beta of 0.1 is desired, input  beta = "01" etc. Make sure that desired laplacian exists in the "Laplacian"- folder.
        Else generate it using the script 'optsolver.m'

    slic_output : bool
        If set, result from SLIC clustering is output.

    seg_output : bool
        If set, result from Ncut segmentation is output.

    slic_output_mode : string
        If set to 'show', shows result from SLIC clustering on screen.
        If set to 'save', saves result from SLIC clustering to a .png file.

    seg_output_mode : string
        If set to 'show', shows result from Ncut segmentation on screen.
        If set to 'save', saves result from Ncut segmentation to a .png file.

    '''

    image_address = "../images/" + imtitle + "." + imtype

    img = Image.open(image_address)
    img = np.array(img)

    isn = init_seg_needed(img, imtitle, compactness, n_segments)
    # isn = True
    # isn = False

    if isn:
        print('Clustering into superpixels...')
        labels = slic(img, compactness=compactness, n_segments=n_segments)
        save_init_seg(labels, img, imtitle, compactness, n_segments)
    else:
        labels = np.load('../tmp/labels/labels_%s.npy' % imtitle)

    g = build_rag(labels, weight_mode, imtitle, beta)

    if weight_mode == 'laplacian':
        max_edge = np.sqrt(g.number_of_nodes())
    else:
        max_edge = 1

    labels2 = graph.cut_normalized(labels, g, thresh = thresh, max_edge = max_edge)

    if not os.path.isdir('../segmentations/%s' % (imtitle)):
        os.mkdir('../segmentations/%s' % (imtitle))

    if slic_output == True:
        output(img, labels, mode = slic_output_mode, filename = '../segmentations/%s/%s_SLIC.png' % (imtitle,imtitle))

    if seg_output == True:
        if weight_mode == 'laplacian':
            output(img, labels2, mode = seg_output_mode, filename = "../segmentations/%s/%s_beta%s_thresh%s.png" % (imtitle,imtitle,beta,str(thresh)))
        elif weight_mode == 'similarity':
            output(img, labels2, mode = seg_output_mode, filename = "../segmentations/%s/%s_classic_thresh%s.png" % (imtitle,imtitle,str(thresh)))


if __name__=='__main__':

    main(imtitle = "road",
    imtype="jpg",
    weight_mode = 'similarity',
    seg_output_mode = 'show',
    thresh = 2e-3,
    beta = "1")
