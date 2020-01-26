import torch
import torchvision

import pandas as pd
import numpy as np
import numpy as np

from functools import partial

# TODO rename jaccard_with_logits
def jaccard(y_true, y_pred):
    """ Jaccard a.k.a IoU score for batch of images
    """
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    
    if y_true_flat.sum() == 0 and y_pred_flat.sum() == 0:
        return torch.tensor(1.0).to(y_true.device)
    
    intersection = (y_true_flat * y_pred_flat).sum(1)
    union = ((y_true_flat + y_pred_flat) > 0.0).float().sum(1)
    
    score = (intersection) / (union + eps)
    score = score.sum() / num
    return score
    
# def jaccard_with_logits(y_true, y_pred):
#     y_pred = torch.argmax(y_pred, dim=1)            
#     return jaccard(y_true, y_pred)


# def jaccard_with_logits_with_borders(y_true, logits):
#     y_pred = torch.argmax(logits, dim=1)
    
#     buildings = (y_true == 1).long()
#     pred_buildings = (y_pred == 1).long()
#     buildings_score = jaccard(buildings, pred_buildings)

#     borders = (y_true == 2).long()
#     pred_borders = (y_pred == 2).long()
#     borders_score = jaccard(borders, pred_borders)
    
#     return torch.stack([buildings_score, borders_score]).mean()


def get_jaccard_with_logits(class_ids):
    
    if isinstance(class_ids, int):
        class_ids = [class_ids]
    
    def jaccard_with_logits(y_true, logits):
        scores = []
        y_pred = torch.argmax(logits, dim=1)
        
        for class_id in class_ids:
            
            scores.append(jaccard(
                (y_true == class_id).long(),
                (y_pred == class_id).long()
            ))

        return torch.stack(scores).mean()
    
    return jaccard_with_logits


# def dice(y_true, y_pred):
#     """ Dice a.k.a f1 score for batch of images
#     """
#     y_pred = torch.argmax(y_pred, dim=1) 

    
#     num = y_true.size(0)
#     eps = 1e-7
    
#     y_true_flat = y_true.view(num, -1)
#     y_pred_flat = y_pred.view(num, -1)
#     intersection = (y_true_flat * y_pred_flat).sum(1)
    
#     score =  (2 * intersection) / (y_true_flat.sum(1) + y_pred_flat.sum(1) + eps)
#     score = score.sum() / num
#     return score


def dice_single_channel(probability, truth, eps = 1e-9):
    p = probability.view(-1).float()
    t = truth.view(-1).float()
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice

def dice(probability, truth):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :])
                mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel

def dice_with_logits(masks, outs):
    """ add threshold param
    """
    return dice((outs.sigmoid() > 0.5).float(), masks)

def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def calc_iou(y_true, y_prediction):
    y_true, y_prediction = map(partial(np.expand_dims, axis=0), (y_true, y_prediction))

    true_objects = len(np.unique(y_true))
    pred_objects = len(np.unique(y_prediction))

    # Compute intersection between all objects
    intersection = np.histogram2d(y_true.flatten(), y_prediction.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(y_true, bins=true_objects)[0]
    area_pred = np.histogram(y_prediction, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    return iou


def calc_score_per_class(y_true, y_prediction):
    iou = calc_iou(y_true, y_prediction)

    # Loop over IoU thresholds
    precisions = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        precisions.append(p)
    return np.mean(precisions)