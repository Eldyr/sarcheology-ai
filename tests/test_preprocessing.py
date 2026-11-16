"""Placeholder module for SARchaeology AI v1.0."""


alignment:
  year: 2024

  reference:
    # Read CRS and resolution from configs/data.yaml.
    source: "study_area"

  output_directory: "data/processed/aligned"
  output_nodata: -9999.0

  quality:
    min_sar_acquisitions: 3
    min_optical_quarters: 2

  resampling:
    continuous_features: "bilinear"
    masks_and_counts: "nearest"

  # Fail rather than producing a dataset with
  # no pixels valid across all three modalities.
  require_joint_coverage: true
