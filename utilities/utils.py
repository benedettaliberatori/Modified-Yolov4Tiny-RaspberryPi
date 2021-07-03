import torch
from collections import Counter



def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=2):
    """
    This function calculates mean average precision (mAP) by evaluating
    the mean of the  average precision for every possible class.
    INPUT:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes but for ground truths
        iou_threshold (float): threshold to evaluate true and false positives
        num_classes (int): number of classes
    """

    # list storing all AP for respective classes
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        
        # Select only the predictions and
        # the ground thruths for class c 
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
                
        # store how many ground thrut bboxes are
        # present in every image, we get a 
        # dictionary like {idx_image : num_bboxes}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        # For every key, val couple we create a tensor 
        # of zeros depending on val
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # we order prediction from most probable to less probable
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # take ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0

            # evaluate the detection against the groung thruth 
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower it means we have a false positive
            else:
                FP[detection_idx] = 1

        # count true and false positives and then
        # both precision and recall
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz to evaluate area under the curve
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)



def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Transform the coordinates of the predictions so that
    to be consistent with the dimension of the entire image.
    INPUT:
        predictions: tensor of size (N, 3, S, S, num_classes+5)
        anchors: anchors used for the predictions
        S: the whole image is idvide into an SxS grid
        is_preds: whether the input is predictions or the true bounding boxes
        
    For predictions we use:
        b_x = sigmoid(t_x) + c_x
        b_y = sigmoid(t_y) + c_y
        b_w = p_w * exp(t_w)
        b_h = p_h * exp(t_h)
    """
    
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[...,1:5]
    
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        # sigmoid for the center coordinates
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        # exponential mutiplied by anchors size for w and h values
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]
        
    # obtain c_x and c_y
    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    
    # add c_x and c_y to center coordinates
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


def non_max_suppression(bboxes, iou_threshold, threshold):
    """
    Compute NMS given bboxes
    INPUT:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
    """

    assert type(bboxes) == list

    # take only bboxes with a certain probability
    bboxes = [box for box in bboxes if box[1] > threshold]
    # sort them
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        # take most probable
        chosen_box = bboxes.pop(0)

        # retain only boxes that do not overlaps
        # too much with the chosen box
        bboxes = [
            box
            for box in bboxes
            if (intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:])
            )
            < iou_threshold)
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    device="cpu",
):
    '''
    Return the bboxes using non max suppression
    INPUT:
        loader : data loader containing image
        model : model that will elaborate images
        iou_threshold : iou thresh. for nms
        anchors : anchors used for predictions
        threshold : second thresholf for nms        
    '''
    
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        # obtain predictions
        for i in range(2):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S

            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[1], anchor, S=S, is_preds=False
        )

        # use nms suprresion over the predictions
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def iou_width_height(boxes1, boxes2):
    '''
    Compute IOU given boxes of which we know height and width
    INPUT:
        boxes1 : first box, we know w and h
        boxes2 : second box, we know w and h
    '''
    
    # take smaller h and smaller w and multiply to obtain intersection
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    
    # sum the area of the two boxes and remove once 
    # the intersection to obtain the union
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(box1, box2):
    '''
    Compute IOU given two boxes of the form (x,y,w,h)
    INPUT:
        boxes_preds : first box
        boxes_labels: second box
    '''
    
    # take coordinates of top left and bottom right
    # corners of first box and second box
    box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
    box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
    box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
    box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
    box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
    box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
    box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
    box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # evaluate interesection and the area of the two boxes
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)



class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    
def class_accuracy(out, model, y, device, threshold=0.5):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for i in range(2):
        y[i] = y[i].to(device)
        obj = y[i][..., 0] == 1 # in paper this is Iobj_i
        noobj = y[i][..., 0] == 0  # in paper this is Iobj_i
        correct_class += torch.sum(
            torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
        )
        tot_class_preds += torch.sum(obj)
        obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
        correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
        tot_obj += torch.sum(obj)
        correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
        tot_noobj += torch.sum(noobj)

    #print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    #print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    #print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()

    return (correct_class/(tot_class_preds+1e-16))*100 ,(correct_obj/(tot_obj+1e-16))*100 , (correct_noobj/(tot_noobj+1e-16))*100


def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"