from msagent import config
import logging

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING= logging.WARNING


class msagentLogger:
    def __init__(self):
        self._logger = logging.getLogger()
        self._formatter = logging.Formatter(("%(asctime)s %(levelname)s %(message)s"))

        sh = logging.StreamHandler()
        sh.setFormatter(self._formatter)
        self._logger.addHandler(sh)

        log_level = config.LOG_LEVEL
        assert log_level in ["debug", "info", "warning", "error"]
        self.set_level(log_level)

        self._prefix = None

    def set_level(self, level):
        assert level in ["debug", "info", "warning"]
        if level == "debug":
            self._logger.setLevel(DEBUG)
        elif level == "info":
            self._logger.setLevel(INFO)
        else:
            self._logger.setLevel(WARNING)

    def set_logfile(self, logdir):
        fh = logging.FileHandler(logdir, mode="w")
        fh.setFormatter(self._formatter)
        self._logger.addHandler(fh)

    def _add_prefix(self, m):
        if self._prefix is None:
            return m
        else:
            return "{} : {}".format(self._prefix, m)

    def set_prefix(self, prefix):
        self._prefix = prefix

    def info(self, m):
        m = self._add_prefix(m)
        self._logger.info(m)

    def debug(self, m):
        m = self._add_prefix(m)
        self._logger.debug(m)

    def warning(self, m):
        m = self._add_prefix(m)
        self._logger.warning(m)

    def error(self, m):
        m = self._add_prefix(m)
        self._logger.error(m)

logger = msagentLogger()
