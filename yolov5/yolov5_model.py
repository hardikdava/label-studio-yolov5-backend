import logging

from label_studio_ml.model import LabelStudioMLBase
import torch
from PIL import Image
from label_studio_ml.utils import get_image_size, get_single_tag_keys, DATA_UNDEFINED_NAME


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


category_map = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}


class YOLOv5Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv5Model, self).__init__(**kwargs)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.model = model
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')

        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)
        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')

        self.score_thresh = 0.4

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        return image_url

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: model returns
            the list of predictions based on input list of tasks
        """
        task = tasks[0]

        predictions = []
        score = 0

        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)


        image = Image.open(image_path)
        original_width, original_height = image.size
        results = self.model(image).pandas().xyxy[0]



        for i, prediction in results.iterrows():
            logging.info(prediction)
            predictions.append({
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'rectanglelabels',
                'score': prediction['confidence'],
                'original_width': original_width,
                'original_height': original_height,
                'image_rotation': 0,
                'value': {
                    "rotation": 0,
                    "x": prediction['xmin'] / original_width * 100,
                    "y": prediction['ymin'] / original_height * 100,
                    "width": (prediction['xmax'] - prediction['xmin']) / original_width * 100,
                    "height": (prediction['ymax'] - prediction['ymin']) / original_height * 100,
                    "rectanglelabels": [category_map[int(prediction.cls.item())]
                }
            })

            score += prediction['confidence']

        return [{
            'result': predictions,
            'score': score / (i + 1),
            'model_version': 'v1',  # all predictions will be differentiated by model version
        }]
