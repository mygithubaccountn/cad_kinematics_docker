"""FreeCAD discovery and STEP import."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pipeline.common.math3d import mat4_identity, mat4_to_list
from pipeline.common.models import (
    AssemblyIR,
    AssemblyNode,
    BBox,
    MateHint,
    MateKind,
    PartInstance,
    Provenance,
)
from pipeline.common.tolerances import Tolerances

_FC_CHECKED = False
_FC_OK = False
_FC_ERROR = ""


def _try_add_freecad_paths() -> None:
    home = Path.home()
    candidates = [
        # macOS (FreeCAD.app bundle)
        Path("/Applications/FreeCAD.app/Contents/Resources/lib"),
        Path("/Applications/FreeCAD.app/Contents/Resources/lib/python3.11/site-packages"),
        home / "Applications/FreeCAD.app/Contents/Resources/lib",
        # Linux (apt/distro packages)
        Path("/usr/lib/freecad/lib"),
        Path("/usr/lib/freecad-python3/lib"),
        Path("/usr/local/lib/freecad/lib"),
        # Linux/macOS (conda-forge — see Dockerfile; also common for manual installs)
        Path("/opt/conda/envs/fc/lib"),
        home / "miniforge3/envs/fc/lib",
        home / "miniconda3/envs/fc/lib",
        home / "anaconda3/envs/fc/lib",
        # Windows (official installer; version dir varies by release)
        Path("C:/Program Files/FreeCAD 1.0/bin"),
        Path("C:/Program Files/FreeCAD 0.21/bin"),
        Path("C:/Program Files/FreeCAD 0.20/bin"),
    ]
    for p in candidates:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.append(str(p))


def freecad_available() -> bool:
    global _FC_CHECKED, _FC_OK, _FC_ERROR
    if _FC_CHECKED:
        return _FC_OK
    _FC_CHECKED = True
    _try_add_freecad_paths()
    try:
        import FreeCAD  # noqa: F401

        _FC_OK = True
        _FC_ERROR = ""
    except Exception as exc:
        _FC_OK = False
        _FC_ERROR = str(exc)
    return _FC_OK


def freecad_status() -> str:
    """Human-readable FreeCAD discovery status."""
    ok = freecad_available()
    if ok:
        import FreeCAD

        ver = ".".join(str(x) for x in FreeCAD.Version()[:3])
        return f"FreeCAD {ver} OK (Python {sys.version.split()[0]})"
    detail = f" ({_FC_ERROR})" if _FC_ERROR else ""
    return (
        f"FreeCAD NOT available under Python {sys.version.split()[0]}{detail}. "
        "Use ./run_with_freecad.sh … (FreeCAD.app bundled Python 3.11)."
    )


def _placement_to_mat4(placement: Any) -> np.ndarray:
    """FreeCAD Placement → 4x4."""
    M = mat4_identity()
    try:
        base = placement.Base
        rot = placement.Rotation
        # FreeCAD Matrix is row-major compatible via toMatrix()
        mat = rot.toMatrix()
        # FreeCAD Matrix A11.. indexing 1-based
        R = np.array(
            [
                [mat.A11, mat.A12, mat.A13],
                [mat.A21, mat.A22, mat.A23],
                [mat.A31, mat.A32, mat.A33],
            ],
            dtype=np.float64,
        )
        M[:3, :3] = R
        M[0, 3] = float(base.x)
        M[1, 3] = float(base.y)
        M[2, 3] = float(base.z)
        # FreeCAD default length unit is mm → convert to metres
        M[:3, 3] *= 1e-3
    except Exception:
        pass
    return M


def _shape_volume_bbox(shape: Any) -> tuple[float, BBox]:
    """Return volume (m^3) and bbox (m) from OCC/FreeCAD shape (mm)."""
    try:
        vol_mm3 = float(shape.Volume)
        box = shape.BoundBox
        if not all(
            np.isfinite(x)
            for x in (vol_mm3, box.XMin, box.XMax, box.YMin, box.YMax, box.ZMin, box.ZMax)
        ):
            return 0.0, BBox(min_xyz=[0, 0, 0], max_xyz=[0, 0, 0])
        bbox = BBox(
            min_xyz=[box.XMin * 1e-3, box.YMin * 1e-3, box.ZMin * 1e-3],
            max_xyz=[box.XMax * 1e-3, box.YMax * 1e-3, box.ZMax * 1e-3],
        )
        return abs(vol_mm3) * 1e-9, bbox
    except Exception:
        return 0.0, BBox(min_xyz=[0, 0, 0], max_xyz=[0, 0, 0])


def _is_plausible_solid(vol: float, bbox: BBox, tol: Tolerances) -> bool:
    if not np.isfinite(vol) or vol < tol.min_part_volume_m3 or vol > tol.max_part_volume_m3:
        return False
    extents = np.asarray(bbox.max_xyz, dtype=np.float64) - np.asarray(bbox.min_xyz, dtype=np.float64)
    if not np.all(np.isfinite(extents)):
        return False
    if float(np.max(extents)) > tol.max_bbox_extent_m:
        return False
    # Reject pure construction planes (near-zero thickness, huge span)
    sorted_ext = np.sort(extents)
    if sorted_ext[0] < 1e-6 and sorted_ext[2] > 1.0:
        return False
    return True


def _dedupe_parts(parts: list[PartInstance]) -> list[PartInstance]:
    """Drop duplicate solids (e.g. assembly compound copies of the same body)."""

    def fingerprint(p: PartInstance) -> tuple:
        bb = p.bbox.as_array()
        # 0.5 mm grid
        q = tuple(int(round(x * 2000.0)) for x in bb.reshape(-1))
        vq = int(round(p.volume * 1e9))
        return q + (vq,)

    def rank(p: PartInstance) -> tuple:
        name = p.name.lower()
        # Prefer explicit part names over exploded assembly copies
        assembly_copy = 1 if "_s" in name or name.startswith("robot_assembly") else 0
        return (assembly_copy, len(name), p.id)

    best: dict[tuple, PartInstance] = {}
    for p in parts:
        fp = fingerprint(p)
        prev = best.get(fp)
        if prev is None or rank(p) < rank(prev):
            best[fp] = p
    return sorted(best.values(), key=lambda p: p.id)


def _iter_solid_objects(doc: Any) -> list[Any]:
    objs = []
    for obj in doc.Objects:
        try:
            # Skip FreeCAD datum / origin / plane objects
            type_id = getattr(obj, "TypeId", "")
            if any(k in type_id for k in ("Plane", "Datum", "Origin", "Line", "Point")):
                continue
            label = (obj.Label or "").lower()
            if "plane" in label and "origin" in type_id.lower():
                continue
            if hasattr(obj, "Shape") and not obj.Shape.isNull():
                sh = obj.Shape
                # Prefer solids / compounds with volume
                if getattr(sh, "Volume", 0) and abs(float(sh.Volume)) > 1e-6:
                    if np.isfinite(float(sh.Volume)):
                        objs.append(obj)
        except Exception:
            continue
    return objs


def _explode_compounds(obj: Any) -> list[tuple[str, Any, Any]]:
    """Return list of (name, shape, placement) solids."""
    results: list[tuple[str, Any, Any]] = []
    shape = obj.Shape
    placement = obj.Placement
    solids = list(getattr(shape, "Solids", []) or [])
    if len(solids) <= 1:
        results.append((obj.Label, shape, placement))
        return results
    for i, solid in enumerate(solids):
        results.append((f"{obj.Label}_s{i}", solid, placement))
    return results


def import_step_freecad(path: Path, tolerances: Tolerances) -> AssemblyIR:
    if not freecad_available():
        raise RuntimeError("FreeCAD not available")

    import FreeCAD
    import Import
    import Part  # noqa: F401

    doc = FreeCAD.newDocument("cad_robot_import")
    try:
        Import.insert(str(path), doc.Name)
        doc.recompute()

        parts: list[PartInstance] = []
        nodes: list[AssemblyNode] = []
        idx = 0
        for obj in _iter_solid_objects(doc):
            for name, shape, placement in _explode_compounds(obj):
                vol, bbox = _shape_volume_bbox(shape)
                if not _is_plausible_solid(vol, bbox, tolerances):
                    continue
                pid = f"part_{idx:04d}"
                idx += 1
                M = _placement_to_mat4(placement)
                parts.append(
                    PartInstance(
                        id=pid,
                        name=name,
                        placement=mat4_to_list(M),
                        volume=vol,
                        bbox=bbox,
                        provenance=Provenance(
                            freecad_path=obj.FullName if hasattr(obj, "FullName") else obj.Name,
                            source="freecad_step",
                        ),
                        shape_ref=obj.Name,
                    )
                )

        parts = _dedupe_parts(parts)
        # Re-assign stable ids after dedupe
        remapped: list[PartInstance] = []
        for i, p in enumerate(sorted(parts, key=lambda x: (x.name, x.id))):
            remapped.append(
                PartInstance(
                    id=f"part_{i:04d}",
                    name=p.name,
                    placement=p.placement,
                    volume=p.volume,
                    bbox=p.bbox,
                    material=p.material,
                    provenance=p.provenance,
                    mesh_vertices=p.mesh_vertices,
                    mesh_faces=p.mesh_faces,
                    shape_ref=p.shape_ref,
                )
            )
        parts = remapped
        nodes = [AssemblyNode(id=f"node_{p.id}", name=p.name, part_id=p.id) for p in parts]

        if not parts:
            raise RuntimeError(f"No solid parts found in STEP: {path}")

        if len(parts) == 1:
            # Policy: warn via meta — fused single solid cannot yield joints
            meta = {
                "warning": "single_solid_assembly",
                "message": "STEP contains a single solid; joint detection requires an assembly of solids.",
            }
        else:
            meta = {"n_parts_raw_before_dedupe": idx}

        mate_hints: list[MateHint] = []
        # Best-effort: FreeCAD Assembly / App::Link constraints if present
        for obj in doc.Objects:
            type_id = getattr(obj, "TypeId", "")
            if "Constraint" in type_id or "Assembly" in type_id:
                mate_hints.append(
                    MateHint(
                        kind=MateKind.UNKNOWN,
                        part_a="",
                        part_b="",
                        confidence=0.2,
                        detail=f"constraint_object:{obj.Label}",
                    )
                )

        return AssemblyIR(
            source_path=str(path.resolve()),
            parts=parts,
            assembly_nodes=nodes,
            mate_hints=mate_hints,
            unit="metre",
            meta={"backend": "freecad", **meta},
        )
    finally:
        try:
            FreeCAD.closeDocument(doc.Name)
        except Exception:
            pass
