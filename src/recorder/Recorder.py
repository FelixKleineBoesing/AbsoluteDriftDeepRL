# TODO embedd pyscreenshot to take screenshots during the game
import pyscreenshot
#TODO embedd keyboard for logging of keyboard actions
import keyboard


class Recorder:
    """
    takes game object and records states and actions
    """

    def __init__(self, save_path: str = "data/records/"):
        self.save_path = save_path

    def record(self):
        pass