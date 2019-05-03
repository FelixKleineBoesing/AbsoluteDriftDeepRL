import numpy as np
from src.preprocessing.Preprocessor import RewardPreprocessor, Preprocessor


class RewardDetector:
    """
    this class uses a minimal neural network to catch the reward from the top left corner
    """
    def __init__(self, preprocessor: Preprocessor):
        assert isinstance(preprocessor, Preprocessor)
        self.preprocessor = preprocessor

    def get_reward(self, img: np.ndarray):
        preprocessed_img = self.preprocessor.preprocess(img)
        reward = 0
        #TODO detect reward from preprocessed image
        return reward