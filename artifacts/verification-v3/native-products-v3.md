# Native raster products v3 verification

**Passed: all 73 native GeoTIFFs from the three fresh confirmation regions; zero failures.** This verifies pixel/grid/mask/provenance correctness, not segmentation accuracy, event accuracy, calibration or geographic transfer.

| Fresh confirmation region | Frames | Native H×W | GeoTIFF bytes | Probability NPY bytes | Registration-excluded frames |
| --- | ---: | --- | ---: | ---: | ---: |
| L15-1172E-1306N_4688_2967_13 | 24 | 1024×1024 | 204,148,647 | 201,326,720 | 0 |
| L15-1204E-1204N_4819_3372_13 | 25 | 1023×1023 | 212,401,749 | 209,305,928 | 0 |
| L15-0358E-1220N_1433_3310_13 | 24 | 1024×1024 | 204,642,034 | 201,326,720 | 0 |

Totals: 621,192,430 GeoTIFF bytes, 611,959,368 probability-NPY bytes, and 306,233,929 original source-image bytes. Verification took 42.30 seconds using CPU2/IO only. 2,518 explicit check results are retained.

Every frame was checked for EPSG CRS, exact cached affine/native dimensions, bounds, float32 band layout/descriptions and NaN nodata. Both probability bands equal `probabilities.npy` exactly on source-valid pixels and become NaN exactly where source-invalid. The validity band, every GDAL band mask, masked reads and the TIFF internal-mask IFD/pixels agree with the original cached source mask. Independent tifffile reads confirm internal one-bit masks; no external `.msk` files are used.

Every original source TIFF was hashed against its recorded SHA, read for exact cached RGB equality, and checked for header dimensions/CRS/affine. Only the already documented <=1e-6-native-unit source-affine roundoff tolerance was accepted, with each actual delta retained. Source-valid pixels never include an invalid source alpha pixel. Source ID/SHA/URI, acquisition interval/date basis, license, publication assumption, model/backbone SHA and the complete tile128/halo64/alignment16/internal2x/inference-source tuple agree across raster tags, prediction.json, sites.json and config v8. Actual spatial/backbone file bytes and current implementation hashes match the recorded identities.

Each probability NPY SHA matches both sites and prediction metadata. All source inputs, originals, products, config and relevant code remained byte-identical across verification. Exact per-frame/file hashes, byte counts, tags, masks/grids and checks are in `native-products-v3-verification.json`; the read-only verifier and full log remain alongside it.

**Mask/coverage scope:** these products preserve source validity (alpha/UDM). Registration-based temporal abstention is separate and must not be encoded as source nodata. The three current fresh regions happen to have zero registration-excluded frames; this does not prove perfect registration or extend that condition to other regions. An observed low footprint probability remains distinct from a masked/no-observation pixel. Internal model interpolation adds no observations or physical resolution.

The cache was explicitly loaded with `labels=False`. The array-read audit contains only `rgb.npy`, `valid.npy` and `probabilities.npy`; no instances, targets, supervision, confirmation event histories or outcome metrics were opened. No model execution, MPS, source/data/model edits, Git actions or public actions occurred.
