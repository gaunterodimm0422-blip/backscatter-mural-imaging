from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import uproot


NUMERIC_BRANCHES = [
    "runID",
    "eventID",
    "edep",
    "time",
    "posX",
    "posY",
    "posZ",
    "sourcePosX",
    "sourcePosY",
    "sourcePosZ",
    "photonID",
    "primaryID",
    "nPhantomCompton",
    "nCrystalCompton",
    "nPhantomRayleigh",
    "nCrystalRayleigh",
]


def _plain_key(key: str) -> str:
    return key.split(";", 1)[0]


def _as_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return _as_python_scalar(value.reshape(-1)[0])
    return value


def _decode_array(values: np.ndarray) -> np.ndarray:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    width = max((len(v) for v in decoded), default=1)
    return np.asarray(decoded, dtype=f"<U{width}")


def _array_mapping(arrays: Any) -> dict[str, np.ndarray]:
    """Normalize uproot TTree dict output and RNTuple structured-array output."""
    if isinstance(arrays, dict):
        return arrays
    names = getattr(getattr(arrays, "dtype", None), "names", None)
    if names:
        return {name: arrays[name] for name in names}
    raise TypeError(f"unsupported uproot array container: {type(arrays)!r}")


def read_gate_metadata(root_path: str | Path) -> dict[str, Any]:
    root_path = Path(root_path)
    metadata: dict[str, Any] = {
        "root_path": str(root_path.resolve()),
        "root_size_bytes": root_path.stat().st_size,
    }
    with uproot.open(root_path) as root:
        names = {_plain_key(key): key for key in root.keys()}
        if "pet_data" in names:
            tree = root[names["pet_data"]]
            arrays = _array_mapping(tree.arrays(library="np"))
            for name, values in arrays.items():
                metadata[name] = _as_python_scalar(values)

        for scalar_name in ("total_nb_primaries", "latest_event_ID"):
            if scalar_name in metadata or scalar_name not in names:
                continue
            obj = root[names[scalar_name]]
            for member in ("fVal", "fValue"):
                try:
                    metadata[scalar_name] = _as_python_scalar(obj.member(member))
                    break
                except Exception:
                    pass

    for key, value in list(metadata.items()):
        if isinstance(value, (np.integer, np.floating)):
            metadata[key] = value.item()

    # GATE 9.4.1 can write the last zero-based event ID into both
    # pet_data.total_nb_primaries and pet_data.latest_event_ID.  Other files
    # write the true count into total_nb_primaries.  Keep both raw values and
    # expose one unambiguous normalization field.
    reported_total = metadata.get("total_nb_primaries")
    latest_event_id = metadata.get("latest_event_ID")
    if latest_event_id is not None:
        event_id_count = int(latest_event_id) + 1
        if reported_total is None or int(reported_total) == int(latest_event_id):
            metadata["primary_event_count"] = event_id_count
            metadata["primary_event_count_source"] = "latest_event_ID_plus_one"
        elif int(reported_total) == event_id_count:
            metadata["primary_event_count"] = int(reported_total)
            metadata["primary_event_count_source"] = "total_nb_primaries"
        else:
            metadata["primary_event_count"] = event_id_count
            metadata["primary_event_count_source"] = (
                "latest_event_ID_plus_one_total_inconsistent"
            )
    elif reported_total is not None:
        metadata["primary_event_count"] = int(reported_total)
        metadata["primary_event_count_source"] = "total_nb_primaries"
    return metadata


def inspect_root(root_path: str | Path) -> dict[str, Any]:
    root_path = Path(root_path)
    report: dict[str, Any] = {
        "path": str(root_path.resolve()),
        "size_bytes": root_path.stat().st_size,
        "metadata": read_gate_metadata(root_path),
        "objects": {},
    }
    with uproot.open(root_path) as root:
        for key in root.keys():
            obj = root[key]
            name = _plain_key(key)
            entry: dict[str, Any] = {"class": obj.classname}
            if hasattr(obj, "num_entries"):
                entry["entries"] = int(obj.num_entries)
                entry["branches"] = list(obj.keys())
            report["objects"][name] = entry
    return report


def _event_key(run_id: np.ndarray, event_id: np.ndarray) -> np.ndarray:
    return (run_id.astype(np.int64) << np.int64(32)) | (
        event_id.astype(np.int64) & np.int64(0xFFFFFFFF)
    )


def _reduce_max(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.maximum.reduceat(values, starts)


def _id_bits(identifier: np.ndarray) -> np.ndarray:
    pid = np.asarray(identifier, dtype=np.int64)
    valid = (pid >= 0) & (pid < 63)
    bits = np.full(pid.shape, np.uint64(1) << np.uint64(63), dtype=np.uint64)
    bits[valid] = np.left_shift(
        np.ones(int(valid.sum()), dtype=np.uint64),
        pid[valid].astype(np.uint64),
    )
    return bits


def _single_id(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint64)
    single = (mask != 0) & ((mask & (mask - np.uint64(1))) == 0)
    single &= (mask & (np.uint64(1) << np.uint64(63))) == 0
    result = np.full(len(mask), -1, dtype=np.int16)
    result[single] = np.rint(np.log2(mask[single].astype(np.float64))).astype(
        np.int16
    )
    return result


def aggregate_hits_tree(root_path: str | Path, tree_name: str) -> dict[str, np.ndarray]:
    """
    Aggregate GATE Hits into one detector record per (runID,eventID).

    GATE output is normally sorted by run/event. A stable sort is applied only
    when that invariant is not satisfied.
    """
    with uproot.open(root_path) as root:
        if tree_name not in root:
            available = ", ".join(_plain_key(k) for k in root.keys())
            raise KeyError(f"missing tree {tree_name}; available: {available}")
        tree = root[tree_name]
        available = set(tree.keys())
        missing = [name for name in ("eventID", "edep", "time", "posX", "posY", "posZ") if name not in available]
        if missing:
            raise KeyError(f"{tree_name} is missing required branches: {missing}")

        branches = [name for name in NUMERIC_BRANCHES if name in available]
        if "comptVolName" in available:
            branches.append("comptVolName")
        arrays = _array_mapping(tree.arrays(branches, library="np"))

    event_id = np.asarray(arrays["eventID"], dtype=np.int64)
    run_id = np.asarray(
        arrays.get("runID", np.zeros(len(event_id), dtype=np.int32)),
        dtype=np.int64,
    )
    key = _event_key(run_id, event_id)
    if len(key) == 0:
        raise ValueError(f"{tree_name} contains no hits")

    if np.any(key[1:] < key[:-1]):
        order = np.argsort(key, kind="stable")
        key = key[order]
        arrays = {name: np.asarray(values)[order] for name, values in arrays.items()}
        event_id = event_id[order]
        run_id = run_id[order]

    starts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    ends = np.r_[starts[1:], len(key)]
    first = starts
    last = ends - 1

    edep_keV = np.maximum(np.asarray(arrays["edep"], dtype=np.float64), 0.0) * 1000.0
    total_energy = np.add.reduceat(edep_keV, starts)
    positive = edep_keV > 0.0

    positions = []
    for branch in ("posX", "posY", "posZ"):
        values = np.asarray(arrays[branch], dtype=np.float64)
        weighted = np.add.reduceat(edep_keV * values, starts)
        centroid = np.divide(
            weighted,
            total_energy,
            out=values[first].copy(),
            where=total_energy > 0.0,
        )
        positions.append(centroid.astype(np.float32))

    hit_time = np.asarray(arrays["time"], dtype=np.float64)
    positive_time = np.where(positive, hit_time, np.inf)
    first_time = np.minimum.reduceat(positive_time, starts)
    fallback_time = np.minimum.reduceat(hit_time, starts)
    first_time = np.where(np.isfinite(first_time), first_time, fallback_time)

    photon_id = np.asarray(
        arrays.get("photonID", np.full(len(key), -1, dtype=np.int16)),
        dtype=np.int64,
    )
    photon_mask = np.bitwise_or.reduceat(_id_bits(photon_id), starts)
    primary_id = np.asarray(
        arrays.get("primaryID", photon_id),
        dtype=np.int64,
    )
    primary_mask = np.bitwise_or.reduceat(_id_bits(primary_id), starts)

    def grouped_max(name: str) -> np.ndarray:
        values = np.asarray(arrays.get(name, np.zeros(len(key))), dtype=np.int32)
        return _reduce_max(values, starts).astype(np.int16)

    if "comptVolName" in arrays:
        compt_volume = _decode_array(np.asarray(arrays["comptVolName"])[last])
    else:
        compt_volume = np.full(len(starts), "UNAVAILABLE", dtype="<U11")

    def first_values(name: str, default: float = 0.0) -> np.ndarray:
        values = np.asarray(
            arrays.get(name, np.full(len(key), default)), dtype=np.float64
        )
        return values[first].astype(np.float32)

    return {
        "key": key[first].astype(np.int64),
        "run_id": run_id[first].astype(np.int32),
        "event_id": event_id[first].astype(np.int32),
        "energy_keV": total_energy.astype(np.float32),
        "x_mm": positions[0],
        "y_mm": positions[1],
        "z_mm": positions[2],
        "time_s": first_time.astype(np.float64),
        "n_hits": (ends - starts).astype(np.int32),
        "photon_mask": photon_mask,
        "photon_id": _single_id(photon_mask),
        "primary_mask": primary_mask,
        "primary_id": _single_id(primary_mask),
        "source_x_mm": first_values("sourcePosX"),
        "source_y_mm": first_values("sourcePosY"),
        "source_z_mm": first_values("sourcePosZ"),
        "n_phantom_compton": grouped_max("nPhantomCompton"),
        "n_crystal_compton": grouped_max("nCrystalCompton"),
        "n_phantom_rayleigh": grouped_max("nPhantomRayleigh"),
        "n_crystal_rayleigh": grouped_max("nCrystalRayleigh"),
        "compt_volume": compt_volume,
    }


def build_pairs(
    d1_events: dict[str, np.ndarray],
    d2_events: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    common, i1, i2 = np.intersect1d(
        d1_events["key"],
        d2_events["key"],
        assume_unique=True,
        return_indices=True,
    )
    if len(common) == 0:
        raise ValueError("D1 and D2 have no common (runID,eventID)")

    output: dict[str, np.ndarray] = {
        "key": common,
        "run_id": d1_events["run_id"][i1],
        "event_id": d1_events["event_id"][i1],
    }
    fields = [
        "energy_keV",
        "x_mm",
        "y_mm",
        "z_mm",
        "time_s",
        "n_hits",
        "photon_mask",
        "photon_id",
        "primary_mask",
        "primary_id",
        "n_phantom_compton",
        "n_crystal_compton",
        "n_phantom_rayleigh",
        "n_crystal_rayleigh",
        "compt_volume",
    ]
    for prefix, table, index in (("d1", d1_events, i1), ("d2", d2_events, i2)):
        for field in fields:
            output[f"{prefix}_{field}"] = table[field][index]

    output["source_x_mm"] = d2_events["source_x_mm"][i2]
    output["source_y_mm"] = d2_events["source_y_mm"][i2]
    output["source_z_mm"] = d2_events["source_z_mm"][i2]
    output["dt_ns"] = (
        output["d1_time_s"].astype(np.float64)
        - output["d2_time_s"].astype(np.float64)
    ) * 1.0e9
    output["different_primary_photons"] = (
        output["d1_primary_mask"] & output["d2_primary_mask"]
    ) == 0
    return output


def save_npz(path: str | Path, arrays: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def pack_event_tables(
    d1_events: dict[str, np.ndarray],
    d2_events: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    packed: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False))
    }
    for prefix, table in (("d1", d1_events), ("d2", d2_events)):
        for name, values in table.items():
            packed[f"{prefix}_{name}"] = values
    return packed
