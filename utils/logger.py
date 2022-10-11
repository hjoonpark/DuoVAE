import os
import datetime
import time
from enum import Enum
"""
Simply prints text with two additional features:
- show timestamp and log level each time
- can be muted
"""

class LogLevel(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3

class Logger():
    def __init__(self, muted=False):
        self.time0 = None
        self.muted = muted

    def print(self, msg: str, level=LogLevel.INFO.name):
        if not self.muted:
            print("{} {} {}".format(self._get_timestamp(), level, msg))

    def _get_timestamp(self):
        now = datetime.datetime.now()
        timestamp = "{:02}-{:02}-{:02} {:02}:{:02}:{:02}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        return timestamp

    def _format_seconds(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return h, m, s
