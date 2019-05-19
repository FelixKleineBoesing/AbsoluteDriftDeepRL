import os
from scipy.misc import imread

from src.detector.RewardDetector import RewardDetector
from src.preprocessing.Preprocessor import RewardPreprocessor

det = RewardDetector(RewardPreprocessor((50, 200)))

def read_and_save_images(image_path: str):
    images = os.listdir(image_path)
    images = [image_path + image for image in images]
    # parse target from image name here and train network
    # image name is ID_Target, example 00005 + _ + 2380X2

    #img = imread(img_path)
    #det.get_reward(img)


def train_reward_detector():
    pass

if __name__=="__main__":
    read_and_save_images("../../data/img/reward_catching/")