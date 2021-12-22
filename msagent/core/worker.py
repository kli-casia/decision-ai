from msagent.utils import service_utils
from msagent.utils import msg_utils
from msagent.utils import logger
from msagent import config as conf

from abc import ABC, abstractmethod
from functools import partial
import types
import zmq


class Worker:

    _entryname2entrypoint = {}
    service_name = None 
    worker_id = None

    def __init__(
        self,
    ):
        assert self.service_name is not None
        assert self.worker_id is not None
        self.prefix = "{}_{}".format(self.service_name, self.worker_id)
        self.counter = service_utils.QPSCounter(self.service_name)

    @classmethod
    def register_entrypoint(cls, entrypoint):
        entry_name = entrypoint.name()
        cls._entryname2entrypoint[entry_name] = entrypoint

    def run(
        self,
    ):
        if conf.USE_PLASMA:
            import pyarrow.plasma as plasma
            self.plasma_client = plasma.connect(service_utils.GetPlasmaDir())

        logger.debug("{} start running ...".format(self.prefix))
        ctx = zmq.Context.instance()

        service_ctrlend = ctx.socket(zmq.REQ)
        service_ctrlend.connect(service_utils.GetServiceCtrlendAddr())
        self.service_ctrlend = service_ctrlend

        event_ctrlend = ctx.socket(zmq.REQ)
        event_ctrlend.connect(service_utils.GetEventCtrlendAddr())
        self.event_ctrlend = event_ctrlend

        servicend = ctx.socket(zmq.DEALER)
        servicend.identity = "{}_{}".format(self.service_name, self.worker_id).encode(
            "ascii"
        )
        servicend.connect(service_utils.GetServiceBackendAddr())
        self.servicend = servicend

        eventend = ctx.socket(zmq.DEALER)
        eventend.identity = "{}_{}".format(self.service_name, self.worker_id).encode(
            "ascii"
        )
        eventend.connect(service_utils.GetEventBackendAddr())
        self.eventend = eventend

        poller = zmq.Poller()
        
        poller.register(servicend, zmq.POLLIN)
        poller.register(eventend, zmq.POLLIN)

        # first register all the entrypoints to router through ctrlend
        logger.info("{}: register entrypoints ...".format(self.prefix))

        self._register_entrypoints_to_router()

        logger.info("{}: start eventloop ...".format(self.prefix))
        self.counter.start()

        # send service ready msg after all the sockets are set
        # which tells that the worker is ready to poll in
        self._send_service_ready_msg()

        # start worker loop
        while True:
            with service_utils.TimeLogger("worker idle"):
                socks = dict(poller.poll())

            if eventend in socks:
                with service_utils.TimeLogger("recv event"):
                    addr, en, bin_msg_id = eventend.recv_multipart()

                with service_utils.TimeLogger(en):
                    if addr == b"RESEND" or not conf.USE_PLASMA:
                        logger.debug("{} recv resend {} msg".format(self.prefix, en))
                        serial_msg = bin_msg_id
                    else:
                        logger.debug("{} recv {} msg".format(self.prefix, en))
                        msg_id = plasma.ObjectID(bin_msg_id)
                        serial_msg = self.plasma_client.get(msg_id)
                    msg = msg_utils.DeSerializeMsg(serial_msg)
                    assert isinstance(msg, msg_utils.EventMsg)

                    self.counter.count(msg)

                    en, payload = msg.event_name, msg.payload

                    assert en in self._entryname2entrypoint
                    entrypoint = self._entryname2entrypoint[en]
                    assert isinstance(entrypoint, EventEntry)
                    entrypoint.handle_message(self, payload)

                    if not addr == b"RESEND":
                        self._send_event_ready_msg()
            
            if servicend in socks:
                with service_utils.TimeLogger("recv service"):
                    addr, en, msg = servicend.recv_multipart()
                    logger.debug("{} recv {} msg".format(self.prefix, en))

                with service_utils.TimeLogger(en, service=True):
                    if conf.USE_PLASMA:
                        msg_id = plasma.ObjectID(msg)
                        msg = self.plasma_client.get(msg_id)
                    else:
                        msg_id = None

                    msg = msg_utils.DeSerializeMsg(msg)
                    assert isinstance(msg, msg_utils.ServiceMsg)

                    self.counter.count(msg)

                    en, args, kwargs = msg.func_name, msg.args, msg.kwargs
                    assert en in self._entryname2entrypoint
                    entrypoint = self._entryname2entrypoint[en]
                    assert isinstance(entrypoint, ServiceEntry)
                    entrypoint.handle_message(self, addr, args, kwargs, msg_id)


    def _register_entrypoints_to_router(self):
        ident = "{}_{}".format(self.service_name, self.worker_id).encode("ascii")
        for en, entrypoint in self._entryname2entrypoint.items():
    
            logger.debug("{}: {} register entrypoint ...".format(self.prefix, en))
            msg = msg_utils.RegisterMsg(
                entry_name=en,
                service_name=self.service_name,
                ident=ident,
                tags=entrypoint.tags,
            )
            serial_msg = msg_utils.SerializeMsg(msg)

            if "event" in entrypoint.tags:
                self.event_ctrlend.send(serial_msg)
                ok = self.event_ctrlend.recv()
                assert ok == b"ok", "register entrypoint fail"
            else:
                self.service_ctrlend.send(serial_msg)
                ok = self.service_ctrlend.recv()
                assert ok == b"ok", "register entrypoint fail"

    def _send_service_ready_msg(
        self,
    ):
        self.servicend.send(b"READY")

    def _send_event_ready_msg(
        self,
    ):
        self.eventend.send(b"READY")


class Entrypoint(ABC):
    def __init__(self, handler):
        self.handler = handler

    @classmethod
    def decorator(cls, *args, **kwargs):
        def registering_decorator(func, args, kwargs):
            args = (func,) + args
            entrypoint = cls(*args, **kwargs)
            Worker.register_entrypoint(entrypoint)
            return func

        if len(args) == 1 and isinstance(args[0], types.FunctionType):
            # usage without arguments to the decorator
            return registering_decorator(args[0], args=(), kwargs={})
        else:
            # usage with arguments to the decorator
            return partial(registering_decorator, args=args, kwargs=kwargs)

    @abstractmethod
    def handle_message(self, args, kwargs):
        pass

    @abstractmethod
    def name(
        self,
    ):
        pass


class ServiceEntry(Entrypoint):
    def __init__(self, handler, batch=1, response=True):
        super().__init__(handler)
        self.batchsize = batch
        self.response = response
        self.tags = ["async"] if response is False else []
        self.buffer = {}

        if conf.USE_PLASMA:
            self.msg_ids = []

    def handle_message(self, worker, addr, args, kwargs, msg_id):
        # TODO: check args and kwargs signiture
        if self.batchsize == 1:
            result = self.handler(worker, *args, **kwargs)
            result = msg_utils.SerializeMsg(result)
            if self.response:
                worker.servicend.send_multipart([addr, result])
                worker._send_service_ready_msg()
            else:
                worker._send_service_ready_msg()
            # delete msg instantly when batchsize = 1
            if conf.USE_PLASMA:
                worker.plasma_client.delete([msg_id])

        elif self.batchsize > 1:
            self.buffer[addr] = args

            if conf.USE_PLASMA:
                self.msg_ids.append(msg_id)

            if len(self.buffer) < self.batchsize:
                worker._send_service_ready_msg()
                return
            else:
                arg_list = list(self.buffer.values())
                addrs = list(self.buffer.keys())
                results = self.handler(worker, arg_list, **kwargs)
                if self.response:
                    assert isinstance(results, list) and len(results) == len(addrs), (
                        "should return a list of results in"
                        "order of the input args list"
                    )
                    for addr, result in zip(list(self.buffer.keys()), results):
                        result = msg_utils.SerializeMsg(result)
                        if conf.USE_PLASMA:
                            worker.servicend.send_multipart([addr, result])
                worker._send_service_ready_msg()
                self.buffer = {}
                # delete msgs when the whole batch has been dealt with
                if conf.USE_PLASMA:
                    worker.plasma_client.delete(self.msg_ids)
                    self.msg_ids = []
        else:
            raise AttributeError("batch size should >= 0")

    def name(
        self,
    ):
        return self.handler.__name__


service = ServiceEntry.decorator


class EventEntry(Entrypoint):
    def __init__(self, handler, event_name):
        super().__init__(handler)
        self.event_name = event_name
        self.tags = ["event"]

    def handle_message(self, worker, payload):
        logger.debug("handling event {}".format(self.event_name))
        self.handler(worker, payload)
        # unlike service entry, event entry don't delete msg
        # since other worker maybe is using it

    def name(
        self,
    ):
        return self.event_name


event_handler = EventEntry.decorator
