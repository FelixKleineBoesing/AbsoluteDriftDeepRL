import pyscreenshot as pysc
import keyboard
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
        managed_dict = mp.Manager().dict()
        managed_dict["running"] = True
        image_process = mp.Process(target=self._record_images, args=(managed_dict, self.save_path))
        key_process = mp.Process(target=self._record_key_strokes, args=(managed_dict, self.save_path))

        image_process.start()
        key_process.start()

        image_process.join()
        key_process.join()

    def _record_images(self, managed_dict, save_path: str):
        try:
            stopped = False
            images = []
            timestamps = []
            while not stopped:
                time_stamp = datetime.datetime.now()
                image = pysc.grab()
                images.append(image)
                timestamps.append(time_stamp)
                print(type(image))
                running = managed_dict["running"]
                print(running)
                print(type(managed_dict["running"]))
                if not bool(managed_dict["running"]):
                    raise KeyboardInterrupt("Execution was interrupted")
        finally:
            if len(images) > 0:
                print(os.getcwd())
                with open(save_path + "images.json", "w") as f:
                    json.dump({"images": images, "timestamps": timestamps}, f)

    def _record_key_strokes(self, managed_dict, save_path: str):
        try:
            records = keyboard.record(until="space+esc")
        finally:
            if len(records) > 0:
                times = []
                keys = []
                for record in records:
                    times.append(record.time)
                    keys.append(record.name)
                key_records = {"keys": keys, "times": times}
                print(os.getcwd())
                with open(save_path + "keys.json", "w") as f:
                    json.dump(key_records,f)


if __name__ == "__main__":
    recorder = Recorder()
    recorder.record()
