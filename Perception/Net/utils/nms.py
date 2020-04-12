import numpy as np


def _iou(box1, box2):
    """
    Code from: https://github.com/mystic123/tensorflow-yolo-v3/blob/136fb66ac4b00e8dfc8391a7ea1c1568a6420206/utils.py#L173

    Computes Intersection over Union value for 2 bounding boxes

    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max(int_x1 - int_x0, 0) * max(int_y1 - int_y0, 0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou






def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    """
    Code from: https://github.com/mystic123/tensorflow-yolo-v3/blob/136fb66ac4b00e8dfc8391a7ea1c1568a6420206/utils.py#L173
    # Applying the correction for issue #70

    Applies Non-max suppression to prediction boxes.
    :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims(
        (predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = {}
    for i, image_pred in enumerate(predictions):
        temp = image_pred
        sum_t = np.sum(temp, axis=1)
        # shape = image_pred.shape
        # non_zero_idxs = np.nonzero(image_pred)
        # image_pred = image_pred[non_zero_idxs]
        # image_pred = image_pred.reshape(-1, shape[-1])
        non_zero_idxs = sum_t != 0
        image_pred = image_pred[non_zero_idxs, :]


        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if cls not in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result