from msagent.core.register import ConsulClient
from msagent.utils import service_utils
from msagent.utils import msg_utils
from msagent.utils import logger
from msagent import config as conf

from functools import wraps
from pyarrow import plasma
import time
import zmq

def recv_in_time(data_socket, timeout=conf.SERVICE_TIMEOUT):
    start_time = time.time()
    while True:
        try:
            result = data_socket.recv(flags=zmq.NOBLOCK)
            return result
        except zmq.ZMQError:
            end_time = time.time()
            duration = end_time - start_time
            if duration > timeout:
                start_time = end_time
                raise TimeoutError("recv timeout")


def get_service(entry_name, is_event=False):
    if conf.IS_DISTRIBUTED:
        consul_client = ConsulClient(conf.CONSUL_PORT)
        logger.debug("client get services ...")
        while True:
            try:
                addr, tags = consul_client.get_service(entry_name, get_all=is_event)
                print("the addr is:  ", addr)
                logger.debug("client get service: {}".format(addr))
                return addr, tags
            except Exception as e:
                logger.error("get service error: {}".format(e))
                time.sleep(3)
    else:
        # in local mode, we ask router to get service info
        context = zmq.Context.instance()
        ctrl_socket = context.socket(zmq.REQ)
        if is_event:
            ctrl_socket.connect(service_utils.GetEventCtrlendAddr())
        else:
            ctrl_socket.connect(service_utils.GetServiceCtrlendAddr())
        msg = msg_utils.DiscoverMsg(entry_name=entry_name)
        serial_msg = msg_utils.SerializeMsg(msg)
        while True:
            try:
                ctrl_socket.send(serial_msg)
                addr, tags = msg_utils.DeSerializeMsg(ctrl_socket.recv())
                if addr:
                    if is_event:
                        return [addr], tags
                    return addr, tags
                else:
                    logger.warning("service not found")
                    time.sleep(1)
            except Exception as e:
                logger.error("get service error: {}".format(e))
                time.sleep(1)


def remote(func):
    addr, tags = get_service(func.__name__)

    context = zmq.Context.instance()
    data_socket = context.socket(zmq.DEALER)
    data_socket.set_hwm(1)
    data_socket.connect(addr)
    response = False if "async" in tags else True

    if not conf.IS_DISTRIBUTED and conf.USE_PLASMA:
        plasma_client = plasma.connect(service_utils.GetPlasmaDir())

    @wraps(func)
    def wrapper(*args, **kwargs):
        with service_utils.TimeLogger("{} remote send".format(func.__name__)):
            msg = msg_utils.ServiceMsg(func.__name__, args, kwargs)
            en = func.__name__.encode("ascii")
            serial_msg = msg_utils.SerializeMsg(msg)

            if conf.IS_DISTRIBUTED or not conf.USE_PLASMA:
                logger.debug("{} send msg".format(en))

                data_socket.send_multipart([en, serial_msg])
            else:
                # if local, use plasma store for comm
                msg_id = plasma_client.put(serial_msg)
                bin_msg_id = msg_id.binary()
                data_socket.send_multipart([en, bin_msg_id])

        if response:
            with service_utils.TimeLogger("{} recv".format(func.__name__)):
                while True:
                    try:
                        result = recv_in_time(data_socket)
                        return msg_utils.DeSerializeMsg(result)
                    except TimeoutError:
                        logger.warning(
                            "client recv msg timeout, " "try to regain service"
                        )
                        addr, _ = get_service(func.__name__)
                        data_socket.connect(addr)
                        logger.info("client reconnect and resend ...")
                        if conf.IS_DISTRIBUTED or not conf.USE_PLASMA:
                            data_socket.send_multipart([en, serial_msg])
                        else:
                            msg_id = plasma_client.put(serial_msg)
                            bin_msg_id = msg_id.binary()
                            data_socket.send_multipart([en, bin_msg_id])

    return wrapper


class EventDispatcher:
    def __init__(
        self,
    ):
        self.context = zmq.Context.instance()
        self.data_socket = self.context.socket(zmq.DEALER)
        if not conf.IS_DISTRIBUTED and conf.USE_PLASMA:
            self.plasma_client = plasma.connect(service_utils.GetPlasmaDir())


    def fire(self, event_name, payload):
        with service_utils.TimeLogger("fire_" + event_name):
            msg = msg_utils.EventMsg(event_name, payload)
            en = event_name.encode("ascii")
            serial_msg = msg_utils.SerializeMsg(msg)
            if conf.IS_DISTRIBUTED or not conf.USE_PLASMA:
                addrs, _ = get_service(event_name, is_event=True)
                for addr in addrs:
                    self.data_socket.connect(addr)
                    self.data_socket.send_multipart([en, serial_msg])
            else:
                # if local, use plasma store for comm
                addrs, _ = get_service(event_name, is_event=True)
                self.data_socket.connect(addrs[0])
                msg_id = self.plasma_client.put(serial_msg)
                bin_msg_id = msg_id.binary()
                self.data_socket.send_multipart([en, bin_msg_id])
