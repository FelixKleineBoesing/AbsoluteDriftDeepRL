import pyscreenshot as pysc
import keyboard
import datetime
import time
import pandas as pd
import multiprocessing as mp


class Recorder:
    """
    takes game object and records states and actions
    """

    def __init__(self, save_path: str = "data/records/"):
        self.save_path = save_path

    def record(self):
        # TODO start image recording in new thread, since keylogging blocks
        queue = mp.Queue()

        pass

    def _record_images(self, queue: mp.Queue):
        # TODO STopped should be set to true if space + Esc is send
        stopped = False
        images = []
        timestamps = []
        while not stopped:
            time_stamp = datetime.datetime.now()
            image = pysc.grab()
            images.append(image)
            timestamps.append(time_stamp)
        queue.put(images)
        queue.put(timestamps)

    def _record_key_strokes(self):
        records = keyboard.record(until="space+esc")
        times = []
        keys = []
        for record in records:
            times.append(record.time)
            keys.append(record.name)
        key_records = pd.DataFrame({"keys": keys, "times": times})

        return key_records

if __name__ == "__main__":
    recorder = Recorder()
    recorder._record_key_strokes()