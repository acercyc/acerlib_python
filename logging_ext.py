# 1.0 - Acer 2017/06/09 15:28
# 2.0 - Acer 2017/12/07 16:46
# 2.1 - Acer 2017/12/21 12:32
import logging
import time
import os


class DataLogger:
    def __init__(self, filename, addHeader=True, mode='w', delimiter=','):
        self.filename = filename
        self.addHeader = addHeader
        self.header = None
        self.count = 0
        self.LastEntry = None
        self.delimiter = delimiter

        # create logger
        logger = logging.getLogger('DataLogger_{}'.format(time.clock()))

        # remove handler if exist
        if logger.hasHandlers():
            logger.handlers = []

        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Add path
        pahtname = os.path.split(filename)[0]
        if not os.path.exists(pahtname) and len(pahtname) > 0:
            os.makedirs(pahtname)

        # add new handler
        h = logging.FileHandler(filename, mode=mode)
        h.setFormatter(logging.Formatter('%(message)s'))
        h.setLevel(logging.DEBUG)
        logger.addHandler(h)

        self.logger = logger

    def log(self, dataDict):
        """
        :param dataDict: a dictionary that indicates the data entry
        :return: Nth record
        """

        if self.count == 0:
            if os.stat(self.filename).st_size == 0 and self.addHeader:
                self.header = list(dataDict.keys())
                self.logger.debug(self._messageFormmating(self.header))
            self.LastEntry = dataDict

        elif self.count > 0:
            self.LastEntry.update(dataDict)

        self.logger.debug(self._messageFormmating(list(self.LastEntry.values())))
        self.count += 1

        return self.count

    def _messageFormmating(self, values):
        return self.delimiter.join([str(x) for x in list(values)])
