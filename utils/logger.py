import os
import datetime
import time
from enum import Enum
"""
Logs to a .txt file
"""
class LogLevel(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3

class Logger():
    """
    simple logger that keeps appending strings to a .txt file
    """
    def __init__(self, save_path):
        self.save_path = save_path
        self.time0 = None
        self.epoch0 = 0
        if os.path.exists(save_path):
            os.remove(save_path)

    def timestamp(self):
        now = datetime.datetime.now()
        timestamp = "{:02}-{:02}-{:02} {:02}:{:02}:{:02}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        return timestamp

    def print(self, msg: str, level=LogLevel.INFO.name):
        timestamp = self.timestamp()
        print("{} {} {}".format(timestamp, level, msg)) # print the message 
        with open(self.save_path, "a") as log_file:
            log_file.write('{} {} {}\n'.format(timestamp, level, msg)) # log the message

    def format_seconds(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return h, m, s
