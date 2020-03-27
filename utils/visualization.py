import copy
from distutils.version import LooseVersion
import math

import cv2
import numpy as np
import scipy.ndimage
import skimage
import skimage.color
import skimage.transform


def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def label_colormap(N=256):
    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def centerize(src, dst_shape, margin_color=None):
    """Centerize image for specified image size

    @param src: image to centerize
    @param dst_shape: image shape (height, width) or (height, width, channel)
    """
    if src.shape[:2] == dst_shape[:2]:
        return src
    centerized = np.zeros(dst_shape, dtype=src.dtype)
    if margin_color:
        centerized[:, :] = margin_color
    pad_vertical, pad_horizontal = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if h < dst_h:
        pad_vertical = (dst_h - h) // 2
    if w < dst_w:
        pad_horizontal = (dst_w - w) // 2
    centerized[pad_vertical:pad_vertical + h,
               pad_horizontal:pad_horizontal + w] = src
    return centerized


def _tile_images(imgs, tile_shape, concatenated_image):
    """Concatenate images whose sizes are same.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param concatenated_image: returned image.
        if it is None, new image will be created.
    """
    y_num, x_num = tile_shape
    one_width = imgs[0].shape[1]
    one_height = imgs[0].shape[0]
    if concatenated_image is None:
        if len(imgs[0].shape) == 3:
            n_channels = imgs[0].shape[2]
            assert all(im.shape[2] == n_channels for im in imgs)
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num, n_channels),
                dtype=np.uint8,
            )
        else:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num), dtype=np.uint8)
    for y in range(y_num):
        for x in range(x_num):
            i = x + y * x_num
            if i >= len(imgs):
                pass
            else:
                concatenated_image[y * one_height:(y + 1) * one_height,
                                   x * one_width:(x + 1) * one_width] = imgs[i]
    return concatenated_image


def get_tile_image(imgs, tile_shape=None, result_img=None, margin_color=None):
    """Concatenate images whose sizes are different.

    Arguments:
        imgs (list[ndarray]): image list to be concatenated
        tile_shape: shape for which images should be concatenated
        result_img: numpy array to put result image
    """
    def resize(*args, **kwargs):
        # anti_aliasing arg cannot be passed to skimage<0.14
        # use LooseVersion to allow 0.14dev.
        if LooseVersion(skimage.__version__) < LooseVersion('0.14'):
            kwargs.pop('anti_aliasing', None)
        return skimage.transform.resize(*args, **kwargs)

    def get_tile_shape(img_num):
        x_num = 0
        y_num = int(math.sqrt(img_num))
        while x_num * y_num < img_num:
            x_num += 1
        return y_num, x_num

    if tile_shape is None:
        tile_shape = get_tile_shape(len(imgs))

    # get max tile size to which each image should be resized
    max_height, max_width = np.inf, np.inf
    for img in imgs:
        max_height = min([max_height, img.shape[0]])
        max_width = min([max_width, img.shape[1]])

    # resize and concatenate images
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        dtype = img.dtype
        h_scale, w_scale = max_height / h, max_width / w
        scale = min([h_scale, w_scale])
        h, w = int(scale * h), int(scale * w)
        img = resize(
            image=img,
            output_shape=(h, w),
            mode='reflect',
            preserve_range=True,
            anti_aliasing=True,
        ).astype(dtype)
        if len(img.shape) == 3:
            img = centerize(img, (max_height, max_width, 3), margin_color)
        else:
            img = centerize(img, (max_height, max_width), margin_color)
        imgs[i] = img
    return _tile_images(imgs, tile_shape, result_img)


def label2rgb(
    lbl, img=None, label_names=None, n_labels=None, ignore_index=-1,
    alpha=0.5, thresh_suppress=0.01
):
    """
    Fill semantic segmentation label with class specific RGB color.
    If RGB image provided, it will overlay label on image.

    Arguments:
        lbl (H×W ndarray): single GT or Prediction label.
        img (H×W×C ndarray) : single original input image.
        label_names (dict or list): names of each label value.
            Key or index is label_value or name of corresponding value.
        n_labels (int): number of classes in label.
        ignore_index (int): value of ignore regions in GT, default is -1.
        alpha (float number between 0 and 1): alpha value used for overlay
            label on image, if `img` provided.
        thresh_suppress (float number between 0 and 1): ignore a label whose
            area proportion under this threshold when putting text.
    """
    if label_names is None:
        if n_labels is None:
            n_labels = lbl.max() + 1  # +1 for bg_label 0
    else:
        if n_labels is None:
            n_labels = len(label_names)
        else:
            assert n_labels == len(label_names)

    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == ignore_index] = (255, 255, 255)  # unlabeled

    if img is not None:
        img_gray = skimage.color.rgb2gray(img)
        img_gray = skimage.color.gray2rgb(img_gray)
        img_gray *= 255
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    if label_names is None:
        return lbl_viz

    # put label names
    np.random.seed(1234)
    for label in np.unique(lbl):
        if label == ignore_index:
            continue

        mask = lbl == label
        if mask.sum() / mask.size < thresh_suppress:
            continue
        mask = (mask * 255).astype(np.uint8)
        y, x = map(int, scipy.ndimage.center_of_mass(mask))

        if lbl[y, x] != label:
            Y, X = np.where(mask)
            point_index = np.random.randint(0, len(Y))
            y, x = Y[point_index], X[point_index]

        text = label_names[label]
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness)

        def get_text_color(color):
            if color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114 > 170:
                return (0, 0, 0)
            return (255, 255, 255)

        color = get_text_color(lbl_viz[y, x])
        cv2.putText(
            lbl_viz, text, (x - text_size[0] // 2, y),
            font_face, font_scale, color, thickness
        )

    return lbl_viz


def visualize_segmentation(
    img, lbl_pred, lbl_true, n_class, label_names=None, ignore_index=-1
):
    """Visualize segmentation.

    Arguments:
        img (H×W×C ndarray): input RGB image.
        lbl_true (H×W ndarray): Ground Truth of the label.
        lbl_pred (H×W ndarray): predicted label.
        n_class (int): number of classes.
        label_names (dict or list): names of each label value.
            Key or index is label_value or name of corresponding value.
        ignore_index (int): value of ignore regions in GT, default is -1.

    Returns:
        img_array (ndarray): visualized image.
        (ignore region will be filled with random colors)
    """

    if lbl_true is None or lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')
    lbl_pred = copy.deepcopy(lbl_pred)
    lbl_true = copy.deepcopy(lbl_true)

    # fill ignore region with random colors
    mask_unlabeled = None
    viz_unlabeled = None
    mask_unlabeled = lbl_true == ignore_index
    lbl_true[mask_unlabeled] = 0
    lbl_pred[mask_unlabeled] = 0
    viz_unlabeled = (
        np.random.random((lbl_true.shape[0], lbl_true.shape[1], 3)) * 255
    ).astype(np.uint8)

    # colorize GT and prediction, and tile them together
    vizs = []
    viz_trues = [
        img,
        label2rgb(lbl_true, label_names=label_names, n_labels=n_class),
        label2rgb(lbl_true, img, label_names=label_names, n_labels=n_class)
    ]
    viz_trues[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
    viz_trues[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
    vizs.append(get_tile_image(viz_trues, (1, 3)))

    viz_preds = [
        img,
        label2rgb(lbl_pred, label_names=label_names, n_labels=n_class),
        label2rgb(lbl_pred, img, label_names=label_names, n_labels=n_class)
    ]
    if mask_unlabeled is not None and viz_unlabeled is not None:
        viz_preds[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        viz_preds[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
    vizs.append(get_tile_image(viz_preds, (1, 3)))

    if len(vizs) == 1:
        return vizs[0]
    elif len(vizs) == 2:
        return get_tile_image(vizs, (2, 1))
    else:
        raise RuntimeError


def visualize_demo(img, prediction, n_class, label_names=None):

    prediction = copy.deepcopy(prediction)
    visualiztion = [
        img,
        label2rgb(prediction, label_names=label_names, n_labels=n_class),
        label2rgb(prediction, img, label_names=label_names, n_labels=n_class)
    ]

    return(get_tile_image(visualiztion, (1, 3)))
