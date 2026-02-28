"""CortexNode — decision publisher (ZMQ PUB with dry-run fallback)."""

import json
import logging

log = logging.getLogger(__name__)


class CortexNode:
    """Publishes SynapticVectors over ZMQ PUB socket.

    If pyzmq is not installed, operates in dry-run mode (logs only).
    """

    def __init__(self, endpoint: str = "tcp://*:5555", dry_run: bool = False):
        self.endpoint = endpoint
        self.dry_run = dry_run
        self._socket = None
        self._context = None

        if not dry_run:
            try:
                import zmq
                self._context = zmq.Context()
                self._socket = self._context.socket(zmq.PUB)
                self._socket.bind(endpoint)
                log.info("CortexNode bound to %s", endpoint)
            except ImportError:
                log.warning("pyzmq not installed — CortexNode in dry-run mode")
                self.dry_run = True

    def publish(self, signal) -> None:
        """Publish a SynapticVector.

        Args:
            signal: SynapticVector instance.
        """
        payload = json.dumps(signal.to_dict())
        if self.dry_run or self._socket is None:
            log.info("[DRY-RUN] %s", payload)
            return
        topic = signal.pair.encode("utf-8")
        self._socket.send_multipart([topic, payload.encode("utf-8")])

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()
