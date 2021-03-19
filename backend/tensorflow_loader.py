# -*- encoding=utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
from backend.utils.nms import single_class_non_max_suppression
from backend.utils.anchor_decode import decode_bbox
from backend.utils.anchor_generator import generate_anchors

target_shape = (260,260)
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
id2class = {0: 'Mask', 1: 'NoMask'}
PATH_TO_TENSORFLOW_MODEL = 'backend/models/face_mask_detection.pb'

def load_tf_model():
    '''
    Load the model.
    :param tf_model_path: model to tensorflow model.
    :return: session and graph
    '''
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_TENSORFLOW_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                sess = tf.Session(graph=detection_graph)
                return sess, detection_graph


def inference(sess, detection_graph, img_arr):

    output_info = []
    height, width, _ = img_arr.shape
    image_resized = cv2.resize(img_arr, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    image_tensor = detection_graph.get_tensor_by_name('data_1:0')
    detection_bboxes = detection_graph.get_tensor_by_name('loc_branch_concat_1/concat:0')
    detection_scores = detection_graph.get_tensor_by_name('cls_branch_concat_1/concat:0')
    # image_np_expanded = np.expand_dims(img_arr, axis=0)
    y_bboxes_output, y_cls_output = sess.run([detection_bboxes, detection_scores],
                                             feed_dict={image_tensor: image_exp})
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=0.5,
                                                 iou_thresh=0.5,
                                                 )
    results = []
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)


        # output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
        results.append({"name": id2class[class_id],
                        "conf": str(conf),
                        "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
                        })

    return {"results": results}

