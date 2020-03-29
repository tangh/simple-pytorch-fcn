import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    # confusion matrix
    mask = (label_true >= 0) & (label_true < n_class)  # ignore region
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2
    ).reshape(n_class, n_class)

    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """
    Arguments:
        label_trues (list[int ndarray (H×W)] OR int ndarray (N×H×W))
        label_preds (list[int ndarray (H×W)] OR int ndarray (N×H×W))
        n_class (int)

    Returns: accuracy score evaluation result (averaged on each image).
        acc (python float number): overall pixel accuracy
        acc_cls (python float number): mean class pixel accuracy
        mean_iou (python float number): mean IoU
        fwavacc (python float number): frequency weighted average iou accuracy
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iou = np.nanmean(iou)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

    return float(acc), float(acc_cls), float(mean_iou), float(fwavacc)


if __name__ == "__main__":
    pred = np.array([
        [0, 0, 1],
        [2, 2, 0],
        [0, 2, 0]
    ], dtype=np.int)
    target = np.array([
        [-1, 0, 3],
        [2, 2, 3],
        [2, 2, 0]
    ], dtype=np.int)
    print(_fast_hist(target.flatten(), pred.flatten(), 4))
