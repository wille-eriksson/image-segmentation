from skimage import color
from PIL import Image
import numpy as np
import os.path

def output(img, labels, mode = "show", filename = "res.png"):
    '''
    Outputs an image based on segmentation labels. Output can be of form save or show.
    '''

    out = color.label2rgb(labels, img, kind='avg')
    result = Image.fromarray(out, mode='RGB')

    if mode == "show":
        result.show()
    elif mode == "save":
        result.save(filename)

def init_seg_needed(img, imtitle, compactness, n_segments):
    '''
    Compares input values for superpixel clustering to the ones of the previous one,
    and determines whether a new one needs to be made.
    '''

    if not os.path.isfile('../tmp/last/last_seg_%s.npy' % (imtitle)):
        return True

    last_seg = np.load('../tmp/last/last_seg_%s.npy' % (imtitle))

    if compactness != last_seg[0]:
        return True

    if n_segments != last_seg[1]:
        return True

    return False

def save_init_seg(labels,img,imtitle,compactness,n_segments):
    '''
    Saves values of the superpixel clustering, so that they can be reused if
    changes are only to be made to the weight assignment.
    '''

    size = np.amax(labels) + 1

    pixel_count = np.zeros(size)
    total_color = np.zeros((size,3))
    mean_color = np.zeros((size,3))
    last_seg = np.zeros(3)

    for index in np.ndindex(labels.shape):
        current = labels[index]
        pixel_count[current] += 1
        total_color[current] += img[index]

    for n in range(size):
        mean_color[n] = (total_color[n] / pixel_count[n])

    last_seg[0] = compactness
    last_seg[1] = n_segments

    np.save('../tmp/labels/labels_%s' % (imtitle), labels)
    np.save('../tmp/regions/regions_%s' % (imtitle), mean_color)
    np.save('../tmp/last/last_seg_%s' % (imtitle), last_seg)


def make_image(w,h,initColor,newColor,name):
    '''
    Creates a two-color image of chosen size.
    '''

    img = Image.new('L',(w,h),color=initColor)

    for i in range(w):
        for j in range(int(h/2)):
            img.putpixel((i,j),newColor)

    img.save(name)

def test_laplacian(Laplacian):
    '''
    Tests constraints for generated laplacian.
    '''

    N = Laplacian.shape[0]

    a = 0
    for i in range(N):
        for j in range(N):
            a += np.absolute(Laplacian[(i,j)]-Laplacian[(j,i)])

    assert a == 0, "Laplacian not symmetric."

    b = 0

    for i in range(N):
        a = 0
        for j in range(N):
            a += Laplacian[(i,j)]
        b += a

    assert np.absolute(b) < 1e-10, "Rows do not add up to one."

    c = 0

    for i in range(N):
        c += Laplacian[(i,i)]

    assert np.absolute(c-N) < 1e-6, "Trace is not N."



    d = 0
    e = 0

    for i in range(N):
        for j in range(N):
            if Laplacian[(i,j)] < 0.00000000000741:
                d += 1
            else:
                e += 1

    assert d > N**2-N, "Too many non-negative values."
