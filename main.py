import numpy as np
from tensorflow import keras
from tqdm import tqdm

from helpers import analyze
from helpers import make_data
from helpers import score_iou
from train import normalization


def post_processing(predictions: np.array) -> np.array:
    """Post processor

    Performs conversions from the model to values expected by the evaluation algorithm.

    Parameters
    ----------
    predictions : np.array
        the predictions made from the model

    Returns
    -------
    np.array
        the predictions after post-processing
    """

    names = ["x", "y", "width", "height", "sin", "cos", "detection"]

    # return nan if no object in image
    if predictions[0][0][0] <= 0:
        array = np.zeros((1, 5))
        array[:] = np.nan
        return array

    # names of the fields
    # fmt: off
    x       = predictions[1][0][0]
    y       = predictions[1][0][1]
    sin     = predictions[2][0][0]
    cos     = predictions[2][0][1]
    width   = predictions[3][0][0]
    height  = predictions[3][0][1]

    # normalization
    x       = normalization(min_x=-1, max_x=1, inputs=x,      tgt_min=10,  tgt_max=190)
    y       = normalization(min_x=-1, max_x=1, inputs=y,      tgt_min=10,  tgt_max=190)
    sin     = normalization(min_x=-1, max_x=1, inputs=sin,    tgt_min=-1,  tgt_max=1)
    cos     = normalization(min_x=-1, max_x=1, inputs=cos,    tgt_min=-1,  tgt_max=1)
    width   = normalization(min_x=-1, max_x=1, inputs=width,  tgt_min=18,  tgt_max=36)
    height  = normalization(min_x=-1, max_x=1, inputs=height, tgt_min=18,  tgt_max=75)
    # fmt: on

    # calculate yaw
    yaw = np.arctan2(sin, cos)
    if yaw < 0:
        yaw += 2 * np.pi

    array = np.asarray(a=[x, y, yaw, width, height])
    array = array.reshape(1, 5)

    return array


def eval():

    # load the proper models for this evaluation
    model = keras.models.load_model("save/best_combined_model")

    ious = []
    analysis = []
    deltas = []

    for _ in tqdm(range(1000)):
        img, label = make_data()

        # perform pre-processing
        img = 2 * img - 1

        predictions = model.predict(img[None])

        # perform post-processing on predictions
        pred = post_processing(predictions)

        pred = np.squeeze(pred)
        ious.append(score_iou(label, pred))

        # analysis tracker
        analysis.append(analyze(label, pred))

        # track the delta
        deltas.append(label - pred)

    ious = np.asarray(ious, dtype="float")
    ious = ious[~np.isnan(ious)]  # remove true negatives
    print((ious > 0.7).mean())

    # statistics
    false_positives = sum(item == "FP" for item in analysis)
    false_negatives = sum(item == "FN" for item in analysis)
    true_negatives = sum(item == "TN" for item in analysis)
    iou_positives = sum(item == "IOU-GOOD" for item in analysis)
    iou_negatvies = sum(item == "IOU-BAD" for item in analysis)

    # display to screen
    print("------Bad Metrics (higher is worse)------")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"IOU Negatives: {iou_negatvies}")
    print("------Good Metrics (higher is better)------")
    print(f"True Negatives: {true_negatives}")
    print(f"IOU Positives: {iou_positives}")


if __name__ == "__main__":
    eval()
