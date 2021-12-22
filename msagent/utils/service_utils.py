from msagent import config as conf
from msagent.utils import logger
import os.path as osp
import platform
import socket
import signal
import time
import sys


def GetServiceBackendAddr(dir=conf.DEFAULT_TMP_DIR, bind=False):
    if platform.system() == "Linux":
        return "ipc://{}/service_backend.ipc".format(dir)
    else:
        port = conf.ROUTER_PORT + 1
        if bind:
            return "tcp://*:{}".format(port)
        return "tcp://127.0.0.1:{}".format(port)


def GetEventBackendAddr(dir=conf.DEFAULT_TMP_DIR, bind=False):
    if platform.system() == "Linux":
        return "ipc://{}/event_backend.ipc".format(dir)
    else:
        port = conf.PUB_PORT + 1
        if bind:
            return "tcp://*:{}".format(port)
        return "tcp://127.0.0.1:{}".format(port)


def GetServiceCtrlendAddr(dir=conf.DEFAULT_TMP_DIR, bind=False):
    if platform.system() == "Linux":
        return "ipc://{}/service_ctrlend.ipc".format(dir)
    else:
        port = conf.ROUTER_PORT + 2
        if bind:
            return "tcp://*:{}".format(port)
        return "tcp://127.0.0.1:{}".format(port)


def GetEventCtrlendAddr(dir=conf.DEFAULT_TMP_DIR, bind=False):
    if platform.system() == "Linux":
        return "ipc://{}/event_ctrlend.ipc".format(dir)
    else:
        port = conf.PUB_PORT + 2
        if bind:
            return "tcp://*:{}".format(port)
        return "tcp://127.0.0.1:{}".format(port)


def GetPubendAddr(dir=conf.DEFAULT_TMP_DIR, bind=False):
    if platform.system() == "Linux":
        return "ipc://{}/pubend.ipc".format(dir)
    else:
        port = conf.PUB_PORT + 3
        if bind:
            return "tcp://*:{}".format(port)
        return "tcp://127.0.0.1:{}".format(port)


def GetLocalIP():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def IsAddressLocal(addr):
    assert addr[:6] == "tcp://"
    addr, port = addr[6:].split(":")
    assert int(port) == conf.ROUTER_PORT
    return addr == "127.0.0.1" or addr == GetLocalIP()


def GetServiceConf(service):
    assert service in conf.SUPPORED_SERVICES, "service not supported"
    service_conf = service.upper() + "_CONFIG"
    try:
        return getattr(conf, service_conf)
    except AttributeError:
        logger.error("service conf not exists")


def GetPlasmaDir():
    return osp.join(conf.DEFAULT_TMP_DIR, "plasma")


class QPSCounter:
    def __init__(self, name, interval=5):
        self.qps = 0
        self.kbs = 0
        self.name = name
        self.interval = interval

        def handler(signum, frame):
            logger.info(
                "{} deal with {:.1f} queries per second".format(
                    self.name, self.qps / interval
                )
            )
            logger.info(
                "{} deal with {:.2f}kb of data per second".format(
                    self.name, self.kbs / interval
                )
            )
            self.qps = 0
            self.kbs = 0
            signal.alarm(interval)

        signal.signal(signal.SIGALRM, handler)

    def count(self, msg):
        self.kbs += sys.getsizeof(msg) / 1000
        self.qps += 1

    def start(
        self,
    ):
        signal.alarm(self.interval)

    def stop(
        self,
    ):
        signal.alarm(0)


class TimeLogger(object):
    def __init__(self, prefix, service=False):
        self.prefix = prefix
        self.check = service

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        time_taken = (self.end - self.start) * 1000
        if self.check:
            if time_taken > conf.SERVICE_TIMEOUT * 1000:
                logger.warning(
                    "{} takes too much time, maybe set service timeout longer".format(
                        self.prefix
                    )
                )
        logger.debug("{} takes time {:5f} ms".format(self.prefix, time_taken))
