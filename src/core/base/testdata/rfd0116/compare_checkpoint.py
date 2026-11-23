"""Compare disabled DefaultDQN checkpoints, exposing libtorch address metadata."""
import hashlib
import io
import pickletools
import struct
import zipfile
from pathlib import Path


def read_checkpoint(path):
    raw = Path(path).read_bytes()
    stream = io.BytesIO(raw)

    def blob():
        size, = struct.unpack("<q", stream.read(8))
        value = stream.read(size)
        if size < 0 or len(value) != size:
            raise ValueError("Invalid archive blob")
        return value

    # intrinsic/skip とも無効の従来形式だけを比較する。
    magic = blob()
    version = stream.read(4)
    header = blob()
    if magic != b"ANET_ARCHIVE" or header != b"DefaultDQNAgent":
        raise ValueError("Expected disabled DefaultDQNAgent checkpoint")
    config = blob()
    networks = (blob(), blob())
    optimizer = blob()
    if stream.read():
        raise ValueError("Unexpected trailing section")
    return raw, (magic, version, header, config), networks, optimizer


def optimizer_records(payload):
    # libtorch は param の C++ アドレスを保存する。params/N で宣言されたもの
    # だけを順序付き識別子へ置換し、tensor/options 等の値はそのまま比較する。
    records = {}
    addresses = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in sorted(archive.namelist()):
            value = archive.read(name)
            if name.endswith("/.data/serialization_id"):
                continue  # 保存内容から生成される libtorch の識別子。
            if name.endswith("/data.pkl"):
                operations = [(op.name, arg) for op, arg, _ in pickletools.genops(value)]
                pending_parameter = None
                for op, arg in operations:
                    if op == "BINUNICODE" and isinstance(arg, str):
                        if arg.startswith("params/"):
                            pending_parameter = arg
                        elif pending_parameter is not None:
                            if not arg.isdecimal():
                                raise ValueError("Expected serialized parameter address")
                            addresses.setdefault(arg, "parameter_" + str(len(addresses)))
                            pending_parameter = None
                value = tuple((op, addresses.get(arg, arg) if isinstance(arg, str) else arg)
                              for op, arg in operations)
            records[name] = value
    return records, len(addresses)


def compare(before, after):
    a = read_checkpoint(before)
    b = read_checkpoint(after)
    left, left_count = optimizer_records(a[3])
    right, right_count = optimizer_records(b[3])
    return {"exact_match": a[0] == b[0],
            "sha256": [hashlib.sha256(item[0]).hexdigest() for item in (a, b)],
            "header_and_config_equal": a[1] == b[1], "network_bytes_equal": a[2] == b[2],
            "optimizer_records_equal": left == right,
            "parameter_address_counts": [left_count, right_count]}
