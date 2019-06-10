import pyscreenshot as pysc
import keyboard
import time
import numpy as np
import datetime
import os
import multiprocessing as mp
import json

lock = mp.Lock()


class Recorder:
    """
    takes game object and records states and actions
    """

    def __init__(self, save_path: str = "../../data/records/"):
        self.save_path = save_path

    def record(self):
        image_process = mp.Process(target=self._record_images, args=(self.save_path, ))
        key_process = mp.Process(target=self._record_key_strokes, args=(self.save_path, ))

        image_process.start()
        key_process.start()

        image_process.join()
        key_process.join()

    @staticmethod
    def _record_images(save_path: str):
        stopped = False
        images = []
        timestamps = []
        index = 0
        while True:
            time_stamp = datetime.datetime.now()
            image = np.array(pysc.grab())
            images.append(image)
            timestamps.append(time_stamp)
            index += 1
            if index % 100 == 0 and index > 0:
                print("{} images saved".format(index))
                with open(save_path + "images.json", "w") as f:
                    json.dump({"images": images, "timestamps": timestamps}, f, cls=NumpyEncoder)

    @staticmethod
    def _record_key_strokes(save_path: str):
        records = keyboard.record(until="space+esc")
        if len(records) > 0:
            times = []
            keys = []
            for record in records:
                times.append(record.time)
                keys.append(record.name)
            key_records = {"keys": keys, "times": times}
            print(os.getcwd())
            with open(save_path + "keys.json", "w") as f:
                json.dump(key_records, f, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    recorder = Recorder()
    recorder.record()
