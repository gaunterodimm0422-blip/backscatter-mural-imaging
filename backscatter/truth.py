from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import uproot


def _array_mapping(arrays: Any) -> dict[str, np.ndarray]:
    if isinstance(arrays, dict):
        return arrays
    names = getattr(getattr(arrays, "dtype", None), "names", None)
    if names:
        return {name: arrays[name] for name in names}
    raise TypeError(f"unsupported uproot array container: {type(arrays)!r}")


def _decode(values: np.ndarray) -> np.ndarray:
    result = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8", errors="replace"))
        else:
            result.append(str(value))
    width = max((len(v) for v in result), default=1)
    return np.asarray(result, dtype=f"<U{width}")


def _event_key(run_id: np.ndarray, event_id: np.ndarray) -> np.ndarray:
    return (run_id.astype(np.int64) << np.int64(32)) | (
        event_id.astype(np.int64) & np.int64(0xFFFFFFFF)
    )


def _find_truth_tree(root: uproot.ReadOnlyDirectory) -> tuple[str, Any]:
    required = {"EventID", "ParentID", "X", "Y", "Z", "CreatorProcess"}
    for key in root.keys():
        obj = root[key]
        if hasattr(obj, "keys") and required.issubset(set(obj.keys())):
            return key.split(";", 1)[0], obj
    candidates = {
        key.split(";", 1)[0]: list(root[key].keys())
        for key in root.keys()
        if hasattr(root[key], "keys")
    }
    raise KeyError(f"no PhaseSpace truth tree with {sorted(required)}; found {candidates}")


def read_compton_truth(truth_path: str | Path) -> dict[str, np.ndarray | str | int]:
    """
    Read recoil-electron vertices produced by Compton interactions.

    Rows are grouped by (run,event,parent gamma); the earliest vertex is retained
    and the number of Compton vertices is stored.
    """
    with uproot.open(truth_path) as root:
        tree_name, tree = _find_truth_tree(root)
        available = set(tree.keys())
        branches = [
            name
            for name in (
                "RunID",
                "EventID",
                "ParentID",
                "TrackID",
                "X",
                "Y",
                "Z",
                "CreatorProcess",
                "ParticleName",
                "ProductionVolume",
                "TimeFromBeginOfEvent",
                "Time",
            )
            if name in available
        ]
        arrays = _array_mapping(tree.arrays(branches, library="np"))

    creator = _decode(np.asarray(arrays["CreatorProcess"]))
    is_compton = np.char.find(np.char.lower(creator), "compt") >= 0
    if "ParticleName" in arrays:
        particle = np.char.lower(_decode(np.asarray(arrays["ParticleName"])))
        is_compton &= np.char.find(particle, "e-") >= 0

    event = np.asarray(arrays["EventID"], dtype=np.int64)[is_compton]
    run = np.asarray(
        arrays.get("RunID", np.zeros(len(creator), dtype=np.int32)),
        dtype=np.int64,
    )[is_compton]
    parent = np.asarray(arrays["ParentID"], dtype=np.int32)[is_compton]
    key = _event_key(run, event)
    x = np.asarray(arrays["X"], dtype=np.float64)[is_compton]
    y = np.asarray(arrays["Y"], dtype=np.float64)[is_compton]
    z = np.asarray(arrays["Z"], dtype=np.float64)[is_compton]

    if "TimeFromBeginOfEvent" in arrays:
        time = np.asarray(arrays["TimeFromBeginOfEvent"], dtype=np.float64)[is_compton]
    elif "Time" in arrays:
        time = np.asarray(arrays["Time"], dtype=np.float64)[is_compton]
    else:
        time = np.arange(len(key), dtype=np.float64)

    if len(key) == 0:
        return {
            "tree_name": tree_name,
            "n_actor_rows": int(len(creator)),
            "n_compton_rows": 0,
            "key": np.empty(0, dtype=np.int64),
            "parent_id": np.empty(0, dtype=np.int32),
            "x_mm": np.empty(0),
            "y_mm": np.empty(0),
            "z_mm": np.empty(0),
            "n_compton_vertices": np.empty(0, dtype=np.int16),
        }

    order = np.lexsort((time, parent, key))
    key = key[order]
    parent = parent[order]
    x, y, z = x[order], y[order], z[order]
    boundary = (key[1:] != key[:-1]) | (parent[1:] != parent[:-1])
    starts = np.r_[0, np.flatnonzero(boundary) + 1]
    ends = np.r_[starts[1:], len(key)]

    return {
        "tree_name": tree_name,
        "n_actor_rows": int(len(creator)),
        "n_compton_rows": int(is_compton.sum()),
        "key": key[starts],
        "parent_id": parent[starts],
        "x_mm": x[starts],
        "y_mm": y[starts],
        "z_mm": z[starts],
        "n_compton_vertices": (ends - starts).astype(np.int16),
    }


def match_truth_to_pairs(
    pairs: dict[str, np.ndarray],
    truth: dict[str, np.ndarray | str | int],
) -> dict[str, np.ndarray]:
    n = len(pairs["key"])
    matched = np.zeros(n, dtype=bool)
    result = {
        "matched": matched,
        "x_mm": np.full(n, np.nan),
        "y_mm": np.full(n, np.nan),
        "z_mm": np.full(n, np.nan),
        "n_compton_vertices": np.zeros(n, dtype=np.int16),
    }
    truth_key = np.asarray(truth["key"], dtype=np.int64)
    if len(truth_key) == 0:
        return result

    # PhaseSpaceActor.ParentID is the Geant4 track ID of the gamma that
    # produced the recoil electron.  In GATE back-to-back sources photonID is
    # a different (and here reversed) label; Hits.primaryID is the matching
    # Geant4 primary track ID.
    pair_parent = np.asarray(
        pairs.get("d1_primary_id", pairs["d1_photon_id"]),
        dtype=np.int32,
    )
    valid_parent = pair_parent >= 0

    dtype = np.dtype([("key", "<i8"), ("parent", "<i4")])
    pair_comp = np.empty(n, dtype=dtype)
    pair_comp["key"] = pairs["key"]
    pair_comp["parent"] = pair_parent
    truth_comp = np.empty(len(truth_key), dtype=dtype)
    truth_comp["key"] = truth_key
    truth_comp["parent"] = np.asarray(truth["parent_id"], dtype=np.int32)

    _, pair_index, truth_index = np.intersect1d(
        pair_comp[valid_parent],
        truth_comp,
        assume_unique=True,
        return_indices=True,
    )
    absolute_pair_index = np.flatnonzero(valid_parent)[pair_index]
    matched[absolute_pair_index] = True
    for name in ("x_mm", "y_mm", "z_mm", "n_compton_vertices"):
        result[name][absolute_pair_index] = np.asarray(truth[name])[truth_index]
    return result
