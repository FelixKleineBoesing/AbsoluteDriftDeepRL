import pyscreenshot as pysc
import keyboard
import logging
import numpy as np
import datetime
import os
import time
import multiprocessing as mp
import json

lock = mp.Lock()


class Recorder:
    """
    takes game object and records states and actions
    """

    def __init__(self, save_path: str = "..\\..\\data\\records\\"):
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
        logging.info("recording images")
        index = 0
        while True:
            time_stamp = datetime.datetime.now()
            image = np.array(pysc.grab())
            index += 1
            if index % 100 == 0:
                print("{} images saved".format(index))
            with open(save_path + r"images\{}.npy".format(str(time_stamp).replace(":", "-")), "wb+") as f:
                np.save(f, image, allow_pickle=False)
            while abs((time_stamp - datetime.datetime.now()).total_seconds()) < 0.4:
                time.sleep(0.05)

    @staticmethod
    def _record_key_strokes(save_path: str):
        logging.info("record key strokes")
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
    recorder = Recorder(save_path="..\\..\\data\\records\\")
    recorder.record()
