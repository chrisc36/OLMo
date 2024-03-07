from os.path import join, exists

from olmo.util import get_bytes_range, read_file


class ObjectStore:
    """Interface for getting objects from a remote store"""

    def get(self, object_id: bytes) -> bytes:
        raise NotImplementedError()

    def contains(self, object_id: bytes) -> bool:
        raise NotImplementedError()

    def write(self, object_id: bytes, data: bytes):
        raise NotImplementedError()


class FileStore(ObjectStore):
    def __init__(self, src_dir):
        self.src_dir = src_dir

    def _id_to_filename(self, object_id: bytes):
        return join(self.src_dir, object_id.hex() + ".bin")

    def contains(self, object_id: bytes) -> bool:
        return exists(self._id_to_filename(object_id))

    def get(self, object_id):
        return read_file(self._id_to_filename(object_id), 0)

    def write(self, object_id, data):
        with open(self._id_to_filename(object_id), "wb") as f:
            f.write(data)


class InMemoryStore(ObjectStore):

    def __init__(self):
        self.store = {}

    def get(self, object_id: bytes) -> bytes:
        return self.store[object_id]

    def contains(self, object_id: bytes) -> bool:
        return object_id in self.store

    def write(self, object_id: bytes, data: bytes):
        self.store[object_id] = data
