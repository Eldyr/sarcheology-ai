"""SARchaeology AI Phase 3.8.1: read-only, pixel-level optical GeoTIFF QA.

Run: uv run python scripts/phase3_optical/validate_optical_exports.py \
    --export outputs/exports/optical_phase38/test_2024

The script never edits GeoTIFFs; QA reports are written to --report-dir.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import array_bounds

STAGES = [(96, 32, 4), (192, 16, 8), (384, 8, 16), (768, 4, 32)]
PATCH = 128
RESIDUAL_PAIRS = [("B11", "B12"), ("B04", "B08"), ("B02", "B03")]


def check_export(export_dir: Path, max_samples: int | None = None) -> dict:
    export_dir = export_dir.resolve()
    manifest_path = export_dir / "export_manifest.json"
    csv_path = export_dir / "patch_embeddings.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    df = pd.read_csv(csv_path)
    errors: list[dict] = []
    warnings: list[dict] = []
    raster_count = 0
    checked = 0
    stats = Counter()

    def issue(kind: str, patch: str, file: str, message: str):
        (errors if kind == "FAIL" else warnings).append(
            {"status": kind, "patch_id": patch, "file": file, "message": message}
        )

    if int(manifest["sample_count"]) != len(df):
        issue("FAIL", "-", str(csv_path), "CSV row count differs from manifest sample_count")
    if df.duplicated(["patch_id", "year", "quarter"]).any():
        issue("FAIL", "-", str(csv_path), "Duplicate patch/year/quarter rows")
    if df["patch_id"].nunique() != int(manifest["usable_geographic_locations"]):
        issue("FAIL", "-", str(csv_path), "Patch ID count differs from manifest")
    if not (df["split"].astype(str) == manifest["split"]).all():
        issue("FAIL", "-", str(csv_path), "Incorrect split in CSV")
    if not (df["year"].astype(int) == int(manifest["year"])).all():
        issue("FAIL", "-", str(csv_path), "Incorrect year in CSV")

    mask_cols = [f"geographic_masked_{i:03d}" for i in range(256)]
    full_cols = [f"geographic_full_{i:03d}" for i in range(256)]
    for name, columns in (("masked", mask_cols), ("full", full_cols)):
        if not set(columns).issubset(df.columns):
            issue("FAIL", "-", str(csv_path), f"Missing {name} embedding columns")
        else:
            v = df[columns].to_numpy(dtype=np.float64)
            if not np.isfinite(v).all():
                issue("FAIL", "-", str(csv_path), f"Nonfinite {name} embeddings")
            elif not np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=0.02):
                issue("WARN", "-", str(csv_path), f"Unexpected {name} embedding norms")

    ref = Affine(*manifest["reference_transform"])
    expected_crs = rasterio.crs.CRS.from_string(manifest["source_crs"])
    min_valid = float(manifest["min_valid_cell_fraction"])
    selected = df.head(max_samples) if max_samples is not None else df

    def read_raster(folder: Path, filename: str, bands: int, side: int,
                    transform: Affine, pid: str):
        nonlocal raster_count
        file = folder / filename
        if not file.is_file():
            issue("FAIL", pid, str(file), "Missing GeoTIFF")
            return None
        try:
            with rasterio.open(file) as src:
                raster_count += 1
                if src.driver != "GTiff":
                    issue("FAIL", pid, str(file), "Driver is not GTiff")
                if (src.count, src.height, src.width) != (bands, side, side):
                    issue("FAIL", pid, str(file),
                          f"Raster shape {(src.count,src.height,src.width)} != {(bands,side,side)}")
                    return None
                if src.crs != expected_crs:
                    issue("FAIL", pid, str(file), f"CRS mismatch: {src.crs}")
                if not np.allclose(tuple(src.transform)[:6], tuple(transform)[:6], atol=1e-7, rtol=0):
                    issue("FAIL", pid, str(file), "Affine transform mismatch")
                if src.nodata is None:
                    issue("FAIL", pid, str(file), "Missing nodata declaration")
                if src.dtypes[0] != "float32":
                    issue("WARN", pid, str(file), f"Expected float32, got {src.dtypes[0]}")
                values = src.read()
                valid = src.dataset_mask() > 0
                if not np.isfinite(values[:, valid]).all():
                    issue("FAIL", pid, str(file), "NaN/Inf inside valid raster pixels")
                if src.nodata is not None and np.any(valid & np.any(values == src.nodata, axis=0)):
                    issue("FAIL", pid, str(file), "Valid raster pixels include nodata sentinel")
                if src.nodata is not None and np.any(~valid & np.any(values != src.nodata, axis=0)):
                    issue("FAIL", pid, str(file), "Masked-out pixels do not contain nodata sentinel")
                return values, valid
        except Exception as exc:
            issue("FAIL", pid, str(file), f"Cannot read raster: {type(exc).__name__}: {exc}")
            return None

    for row in selected.itertuples(index=False):
        checked += 1
        pid = str(row.patch_id)
        relative_folder = Path(str(row.folder).replace('\\', '/'))
        if relative_folder.is_absolute() or '..' in relative_folder.parts:
            issue("FAIL", pid, str(row.folder), "Unsafe folder path")
            continue
        folder = export_dir / relative_folder
        expected_transform = ref * Affine.translation(int(row.col), int(row.row))
        original = read_raster(folder, "valid_pixels_10m.tif", 1, 128,
                               expected_transform, pid)
        if original is None:
            continue
        image, _ = original
        raw = image[0]
        if not np.isin(raw, [0.0, 1.0]).all():
            issue("FAIL", pid, "valid_pixels_10m.tif", "Source validity values must be 0 or 1")
        valid_10m = raw > 0.5
        frac = float(valid_10m.mean())
        if not np.isclose(frac, float(row.valid_fraction), atol=1e-6):
            issue("FAIL", pid, "valid_pixels_10m.tif", "Valid fraction differs from CSV")
        if frac < 0.5:
            issue("WARN", pid, "valid_pixels_10m.tif", "Source patch has under 50% valid pixels")

        bounds = array_bounds(PATCH, PATCH, expected_transform)
        given_bounds = [row.left, row.bottom, row.right, row.top]
        if not np.allclose(bounds, given_bounds, atol=1e-5, rtol=0):
            issue("FAIL", pid, str(folder), "CSV bounds mismatch source patch transform")

        for stage, (channels, side, stride) in enumerate(STAGES):
            coarse_transform = expected_transform * Affine.scale(stride, stride)
            cell_frac = valid_10m.reshape(side, stride, side, stride).mean(axis=(1, 3))
            expected_mask = cell_frac >= min_valid
            fractions = read_raster(folder, f"swin_stage{stage}_valid_fraction.tif",
                                    1, side, coarse_transform, pid)
            if fractions is not None:
                vals, masks = fractions
                if not masks.all():
                    issue("FAIL", pid, str(folder), f"Stage {stage} fraction raster has unexpected masked cells")
                if not np.allclose(vals[0], cell_frac, atol=1e-6, rtol=0):
                    issue("FAIL", pid, str(folder), f"Stage {stage} validity fractions disagree with source mask")
            features = read_raster(folder, f"swin_stage{stage}_unmasked.tif",
                                   channels, side, coarse_transform, pid)
            if features is not None:
                values, mask = features
                if not np.array_equal(mask, expected_mask):
                    issue("FAIL", pid, str(folder), f"Stage {stage} GDAL mask disagrees with thresholded validity")
                if mask.any():
                    stats[f"stage{stage}_valid_values"] += int(mask.sum() * channels)
                    feature_std = np.std(values[:, mask].astype(np.float64), axis=1)
                    if (feature_std < 1e-9).mean() > 0.95 and mask.sum() > 16:
                        issue("WARN", pid, str(folder), f"Stage {stage} >95% channels are spatially constant")
                    if np.max(np.abs(values[:, mask])) > 1e5:
                        issue("WARN", pid, str(folder), f"Stage {stage} feature magnitude exceeds 1e5")

        if bool(manifest.get("spectral_residuals_enabled")):
            for pair in manifest.get("spectral_residual_pairs", RESIDUAL_PAIRS):
                filename = f"spectral_residual_{'_'.join(pair)}_signed_reflectance.tif"
                residual = read_raster(folder, filename, 2, 128, expected_transform, pid)
                if residual is not None:
                    vals, mask = residual
                    if not np.array_equal(mask, valid_10m):
                        issue("FAIL", pid, filename, "Spectral residual GDAL mask differs from source validity")
                    if mask.any():
                        z = vals[:, mask]
                        stats["residual_valid_pixels"] += int(z.size)
                        if np.max(np.abs(z)) > 1.0:
                            issue("WARN", pid, filename, "Absolute signed residual exceeds reflectance unit 1.0")

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "phase": "3.8.1", "status": status,
        "export_directory": str(export_dir), "split": manifest["split"],
        "year": manifest["year"], "manifest_samples": int(manifest["sample_count"]),
        "checked_samples": checked, "checked_rasters": raster_count,
        "counts": {"errors": len(errors), "warnings": len(warnings)},
        "statistics": dict(stats), "errors": errors, "warnings": warnings,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True,
                   help="Folder containing export_manifest.json and patch_embeddings.csv")
    p.add_argument("--report-dir", type=Path, default=None,
                   help="Separate QA output folder (default: outputs/quality/optical_phase38/<export-name>)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Optional fast audit on first N rows; omit to audit all rasters")
    args = p.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        p.error("--max-samples must be positive")
    output = args.report_dir or (Path("outputs/quality/optical_phase38") / args.export.name)
    if output.resolve() == args.export.resolve() or args.export.resolve() in output.resolve().parents:
        p.error("Report directory must not be inside the export directory")
    result = check_export(args.export, args.max_samples)
    output.mkdir(parents=True, exist_ok=True)
    (output / "geotiff_qa_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(result["errors"] + result["warnings"],
                 columns=["status", "patch_id", "file", "message"]).to_csv(
                     output / "geotiff_qa_issues.csv", index=False)
    pd.DataFrame([{k: v for k, v in result.items() if k not in ("errors", "warnings", "statistics")}]).to_csv(
        output / "geotiff_qa_summary.csv", index=False)
    print(f"Status: {result['status']} | {result['checked_samples']} patches | "
          f"{result['checked_rasters']} rasters | {result['counts']['errors']} errors | "
          f"{result['counts']['warnings']} warnings")
    print("Report:", output / "geotiff_qa_report.json")
    raise SystemExit(2 if result["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
