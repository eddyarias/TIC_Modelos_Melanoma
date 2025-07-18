import os
import cv2
import numpy as np
from openvino.runtime import Core 
from scipy.spatial import distance as scipy_distance

class Siamese_OpenVino(object):
    """FakeID abstract base class."""

    def __init__(
        self,
        model_path='./checkpoints/20240318_211856_AWS/frozen_model.xml',
        distance='euclidean'
    ):
        """Class instantiation.

        Parameters
        ----------
        model_path : str
            Path to xml model
        distance : str
            Distance function < euclidean | cosine >
        """

        self.model_path = model_path
        self.distance = distance

        # Load model
        ie = Core()
        siamese_model = ie.read_model(model=self.model_path)
        self.siamese_model = ie.compile_model(model=siamese_model, device_name="CPU")

        # Get input shape
        input_layer = self.siamese_model.input(0) # input layer info
        _, C, H, W = input_layer.shape # input shape
        self.target_size = (W, H)

        # Load templates
        temp_path = os.path.dirname(self.model_path) + '/templates.npz'
        self.templates = np.load(temp_path)["templates"]

        print('Siamese Network initilized on OpenVINO')

    def process_image(self, image):
        """Transforms the input image to tensor format.

        Inputs
        ----------
        image : numpy array
            Input image.
        
        Outputs
        ----------
        tensor : numpy array
            Output tensor with the right dimensions.
        """

        # Resize Image
        if image.shape[:2] != self.target_size[::-1]:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Convert to tensor
        tensor = np.expand_dims(np.transpose(image, (2,0,1)), 0)
        return tensor
    
    def compute_dist(self, result):
        """Computes distance between the feature vector and 4 templates.

        Inputs
        ----------
        result : numpy array
            Backbone's feature-vector prediction.
        
        Outputs
        ----------
        dist : float
            Average distance to the 4 templates in the feature space.
        """
        if self.distance in ['euclidean','Euclidean','EUCLIDEAN','euc']:
            dist = np.linalg.norm(result - self.templates, axis=1)
        elif self.distance in ['cosine','Cosine','COSINE','cos']:
            dist = scipy_distance.cdist(result, self.templates, metric='cosine')
            if np.linalg.norm(result) < 0.00000000001:  # Case when the CNN generates a vector that is all 0's
                dist = np.ones(dist.shape)
        else:  # Euclidean distance by default
            dist = np.linalg.norm(result - self.templates, axis=1)
        return dist.mean()

    def predict(self, image):
        """Makes a prediction regarding an input Pillow Image.

        Inputs
        ----------
        image : numpy array
            Input image.
        
        Outputs
        ----------
        score : float
            Model's prediction of the bona-fide likelyhood. Value between 0 and 1.
        """
        # Makes a prediction
        tensor = self.process_image(image)
        result = self.siamese_model(tensor)[0]
        dist = self.compute_dist(result)
        score = 1 - dist/1.5
        return score
    
    def __str__(self):
        return (
            f"Weights: {self.model_path}\n"
            f"Target size: {self.target_size}\n"
            f"Distance: {self.distance}\n"
        )

