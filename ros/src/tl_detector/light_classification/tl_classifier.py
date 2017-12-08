import hashlib
import os
import sys
import tarfile
import urllib

import numpy as np
import tensorflow as tf

import rospy
from styx_msgs.msg import TrafficLight

sys.path.append(os.path.dirname(__file__))
from object_detection.utils import label_map_util  # isort:skip

THRESHOLD = 0.5
MODEL_URL = 'https://github.com/TeenageMutantNinjaTurtlesAllTheWayDown/CarND-Capstone/releases/download/model/model.tar'
MODEL_SHA256 = '255bdda11f7fe7f72340bfec0ce3fc2db7f6d941368b78281b64ce2420142a0d'


def download_model(variant):
    dir = os.path.dirname(__file__)
    archive_path = os.path.join(dir, 'model.tar')
    model_dir = os.path.join(dir, 'model')
    graph_path = os.path.join(model_dir, variant, 'graph.pb')
    label_map_path = os.path.join(model_dir, variant, 'label_map.pbtxt')

    if os.path.exists(graph_path):
        return graph_path, label_map_path

    rospy.logwarn('Downloading model...')

    for i in range(3):
        urllib.urlretrieve(MODEL_URL, archive_path)
        digest = hashlib.sha256(open(archive_path, 'rb').read()).hexdigest()
        if digest != MODEL_SHA256:
            os.remove(archive_path)
        else:
            break

    if not os.path.exists(archive_path):
        raise Exception('Wrong traffic light detector model checksum')

    with tarfile.open(archive_path) as file:
        file.extractall(dir)

    os.remove(archive_path)

    rospy.logwarn('Model was successfully downloaded')

    return graph_path, label_map_path


class TLClassifier(object):
    def __init__(self, variant, width, height):
        graph_path, label_map_path = download_model(variant)
        rospy.logwarn(graph_path)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        with self.sess.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            graph = self.sess.graph

            self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
            self.detection_scores = graph.get_tensor_by_name(
                'detection_scores:0')
            self.detection_classes = graph.get_tensor_by_name(
                'detection_classes:0')

            fake_img = np.zeros((1, height, width, 3), np.uint8)
            self.sess.run([self.detection_scores, self.detection_classes],
                          feed_dict={self.image_tensor: fake_img})

        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=4, use_display_name=True)
        self.categories = {}
        for category in categories:
            self.categories[category['id']] = category['name']

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = np.asarray(image, np.uint8)
        image = np.expand_dims(image, axis=0)

        tl = None

        scores, classes = self.sess.run([self.detection_scores, self.detection_classes],
                                        feed_dict={self.image_tensor: image})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int)

        for i in range(len(scores)):
            if scores[i] < THRESHOLD:
                continue

            category = self.categories[classes[i]]
            if tl is None:
                tl = category
            elif category != tl:
                return TrafficLight.UNKNOWN

        if tl == 'Red':
            return TrafficLight.RED
        elif tl == 'Yellow':
            return TrafficLight.YELLOW
        elif tl == 'Green':
            return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
