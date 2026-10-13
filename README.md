# ImageSleuth AI
### Intelligent Digital Image Forensics and Tampering Analysis System

> **Educational forensic-assistance tool.** ImageSleuth AI surfaces *indicators* that a human examiner can
> investigate. It **cannot** conclusively determine whether an image is authentic or manipulated, gives no
> "real/fake" verdict and no confidence percentage, does not maintain a formal chain of custody, and is not
> certified for legal use.

---

## 1. Problem statement

Editing images is now easy, and edited images spread quickly. Investigators, journalists and students need
a transparent way to examine an image file. That means checking its fingerprint and metadata and running
well-known forensic techniques whose behaviour they can understand and explain, without uploading
sensitive images to an online service.

## 2. Objectives

1. Accept JPEG/PNG images safely and validate them by content, not just by extension.
2. Record a SHA-256 fingerprint of the exact uploaded bytes **before** any analysis, and allow comparison
   against a trusted reference hash.
3. Extract and explain EXIF metadata, treating missing metadata as normal rather than suspicious.
4. Implement Error Level Analysis (ELA) and keypoint-based copy-move analysis that are genuine,
   reproducible and explainable.
5. Provide exploratory consistency views (noise, edges, channels).
6. Present every result with its parameters and limitations, and produce a downloadable PDF report.
7. Run fully offline on a normal laptop: no GPU, no API keys, no cloud.

## 3. Features

| Area | What it does |
|---|---|
| **Evidence intake** | Drag-and-drop upload (JPEG/JPG/PNG, max 20 MB, max 50 MP). Validation by magic bytes plus Pillow `verify()` plus a full decode. Corrupt, truncated and renamed files are rejected with a clear message. Each upload gets a unique investigation ID (`ISA-YYYYMMDD-HHMMSS-XXXXXXXX`). |
| **Preview and file info** | Image preview, dimensions, megapixels, format (detected from content), file size, colour mode, upload time. |
| **Metadata** | Camera make/model, original/digitised/modified timestamps, software, orientation, and GPS (hidden by default). Full EXIF table and container info (JFIF/PNG chunks/ICC/XMP). Editing-software tags and timestamp differences are reported as *weak indicators*; missing EXIF is reported as a *neutral observation*. |
| **Integrity** | SHA-256 over the exact uploaded bytes, computed at intake. Constant-time comparison with a user-supplied reference hash. Re-verification that the bytes are unchanged, on screen and at report time. |
| **Error Level Analysis** | In-memory JPEG recompression at a configurable quality, an amplified difference image, a heat map, a tile-level luma ELA-to-texture outlier heuristic, and a "JPEG ghost" quality sweep. PNG input is clearly labelled as an experiment. |
| **Copy-move analysis** | ORB keypoints, k-NN self-matching, the g2NN ratio test, a distance filter and iterative RANSAC similarity-transform clustering. Output includes visual duplicated-region pairs and a table of shift, rotation and scale. |
| **Consistency tools** | Noise residual, local noise-level map, Canny edges, R/G/B and Y/Cr/Cb channel views. All are labelled as indicators. |
| **Results dashboard** | Three-state summary: *Potential indicators found*, *No indicators detected by implemented checks* or *Analysis incomplete*. Shows indicators, neutral observations, check status and visual evidence. |
| **PDF report** | Investigation ID, timestamps, filename, file properties, SHA-256 with re-verification, metadata summary (GPS redacted unless you opt in), tests and parameters, visualisations, findings split into indicators, observations and conclusions, limitations, evidence-handling notes and a disclaimer. Built in memory. |
| **Privacy** | Runs on localhost. No third-party APIs. No image contents in logs. Nothing is saved permanently. Streamlit telemetry is disabled. |

## 4. Architecture

```
ImageSleuth-AI/
├── app.py                         Streamlit UI (presentation only)
├── forensic/                      UI-independent, unit-tested analysis package
│   ├── case.py                    Evidence intake: hash FIRST, validate, decode read-only copy, metadata
│   ├── validation.py              Size/extension/magic-byte/decoder checks, safe decoding
│   ├── integrity.py               SHA-256, reference comparison, re-verification
│   ├── metadata.py                EXIF (IFD0, Exif IFD, GPS IFD), GPS conversion and redaction
│   ├── ela.py                     Error Level Analysis, tile outliers, quality sweep
│   ├── copy_move.py               ORB + g2NN + RANSAC copy-move analysis
│   ├── visualizations.py          Noise residual, noise map, edges, channel views
│   ├── summary.py                 Rule-based three-state summary (no scores)
│   ├── reporting.py               ReportLab PDF generation (in memory)
│   └── utils.py                   IDs, timestamps, conversions
├── scripts/generate_demo_images.py  Synthetic demo images plus a ground-truth manifest
├── tests/                         Software-behaviour tests (pytest)
│   └── synthetic_benchmarks/      Synthetic sanity checks (kept separate, marked "synthetic")
├── assets/                        Demo images are generated into assets/demo/
├── .streamlit/config.toml         Dark theme, upload limit, localhost binding, telemetry off
├── requirements.txt               Pinned dependencies
└── pytest.ini
```

**Data flow**

```
upload bytes ─► SHA-256 (immutable bytes) ─► validation ─► decoded RGB copy (read-only)
                                                   │
                     ┌──────────────┬──────────────┼──────────────┬───────────────┐
                  metadata         ELA        copy-move     consistency      reference hash
                     └──────────────┴──────┬───────┴──────────────┴───────────────┘
                                     summary (3 states)
                                           │
                       dashboard  +  PDF report (in memory, re-verifies hash)
```

The UI and the analysis are kept separate on purpose. Every forensic function can be tested with
`pytest` without a browser, and the UI only calls those functions and displays their results.

## 5. Algorithms explained

### 5.1 SHA-256 integrity
SHA-256 maps any byte sequence to a 256-bit digest. Changing a single bit changes about half the
output bits (the avalanche effect), and finding two inputs with the same digest is computationally
infeasible. ImageSleuth hashes the **exact uploaded bytes** before anything else touches them.
**Limitation:** a matching hash proves only *byte-level equality with whatever produced the reference*.
A manipulated image has a perfectly valid hash of its own, and an innocent re-save produces a different hash.

### 5.2 Metadata (EXIF)
Pillow reads IFD0 (make, model, software, DateTime, orientation), the Exif sub-IFD (DateTimeOriginal,
DateTimeDigitized) and the GPS sub-IFD (degrees/minutes/seconds, converted to signed decimal degrees).
Weak indicators are raised for (a) a Software tag naming an image editor and (b) DateTime ≠ DateTimeOriginal.
Missing EXIF is **never** treated as an indicator.

### 5.3 Error Level Analysis (ELA)
1. Re-encode the decoded pixels as JPEG at quality *Q* (default 90, 4:2:0 subsampling, in memory).
2. Compute |original − recompressed| per pixel and amplify it, either automatically (99.5th percentile
   mapped to full brightness) or by a fixed factor.
3. **Tile heuristic.** Using luma (Y) only, compute for each 32×32 textured tile
   `ratio = log((mean ELA + 0.5) / (mean Sobel gradient + 1))`. Turn the ratios into robust z-scores
   (median/MAD). A tile is flagged if z > 4, its mean error is ≥ 1.5, **and** at least one neighbouring tile
   is also above threshold. Dividing by texture stops ordinary detail from being flagged just for being
   busy. Luma-only avoids colour-edge artefacts from chroma subsampling. The neighbour rule suppresses
   isolated noise.
4. **Quality sweep ("JPEG ghost", simplified from Farid 2009).** Recompress at Q = 60…95 and show
   normalised, locally averaged squared luma differences. A region previously compressed at quality q turns
   dark at q.

**Why it's not proof:** edges and texture are always bright. PNGs, screenshots, resized and repeatedly
saved images are hard to interpret. The heuristic is uncalibrated. ELA is most sensitive when *Q* is close
to the quality at which part of the image was previously saved.

### 5.4 Copy-move analysis
1. **ORB** (Oriented FAST + Rotated BRIEF) finds up to 5,000 corner-like keypoints with 256-bit binary
   descriptors. The image is downscaled to at most 1,600 px for speed, and coordinates are mapped back.
2. **Self-matching:** brute-force Hamming k-NN (k = 3) of the image against itself. The trivial self-match
   is discarded.
3. **g2NN ratio test:** keep a match only if `best < 0.75 × second-best`. Also require a descriptor
   distance ≤ 48 bits and a spatial distance ≥ 40 px.
4. **Geometric verification:** both directions of every pair go into iterative **RANSAC** fitting of a
   similarity transform (`cv2.estimateAffinePartial2D`). Clusters with ≥ 8 inliers and a plausible
   scale (0.5–2) become candidate duplicated-region pairs. Their pairs are removed, and the search repeats
   (up to 5 clusters). The RNG is seeded, so results are reproducible.

**Limitations:** smooth regions have no keypoints. Repetitive real-world structures produce consistent
matches without any forgery (demo image 05 shows this deliberately). Mirrored copies are not modelled.
Splicing from *another* image is out of scope.

### 5.5 Consistency tools
* **Noise residual:** |gray − median3×3(gray)| × amplification.
* **Noise-level map:** per-block standard deviation of the residual, shown as a heat map.
* **Canny edges:** thresholds of median × (1 ± σ).
* **Channels:** R, G, B and Y, Cr, Cb.
These are visual aids only. Focus, lighting, depth of field and denoising also change noise and edges.

### 5.6 Overall summary
A rule-based summary with three states. *Potential indicators found* if any check produced an indicator.
Otherwise *Analysis incomplete* if a core check did not complete. Otherwise *No indicators detected by
implemented checks*, which explicitly does **not** mean "authentic". There is no score, because no
validated, calibrated model exists for these checks.

## 6. Installation

Requirements: **Python 3.10–3.12** (developed and tested on 3.11) and about 600 MB of disk space for the
dependencies. No GPU, Docker, database or API keys are needed.

### Windows (PowerShell)
```powershell
cd ImageSleuth-AI
py -3.11 -m venv .venv                       # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\generate_demo_images.py       # optional: synthetic demo images
streamlit run app.py                         # opens http://localhost:8501
```

### macOS / Linux
```bash
cd ImageSleuth-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/generate_demo_images.py       # optional
streamlit run app.py
```

After installation, everything works **offline**.

## 7. Running tests

```bash
pytest                         # everything (software tests + synthetic sanity checks + UI smoke tests)
pytest -m "not synthetic"      # only software-behaviour tests
pytest -m synthetic            # only the synthetic forensic sanity checks
pytest -m app                  # only the headless Streamlit UI smoke tests
pytest --cov=forensic          # with coverage (pytest-cov, optional extra)
```

**Two kinds of tests, kept separate on purpose:**

* **Software-behaviour tests** (`tests/test_*.py`) check that the code does what it claims. They cover
  hash test vectors, rejection of malformed files, EXIF parsing, GPS redaction, ELA determinism and
  parameters, copy-move result structure, PDF validity and escaping, and that evidence bytes stay unchanged.
  `test_app_smoke.py` runs the Streamlit UI headlessly with `streamlit.testing`.
* **Synthetic sanity checks** (`tests/synthetic_benchmarks/`, marker `synthetic`) confirm that each
  detector responds to the edit it was designed for on *computer-generated* images with known edits, and
  that it stays quiet on the unedited version. They are **not** accuracy measurements. Real accuracy would
  need a labelled real-world dataset (such as CASIA v2, Columbia, CoMoFoD or MICC-F220) and a proper evaluation.

## 8. Demo images and demo workflow

```bash
python scripts/generate_demo_images.py        # writes assets/demo/ + manifest.json (ground truth)
```

| File | What it demonstrates |
|---|---|
| `01_synthetic_original.jpg` | Baseline: synthetic EXIF, no indicators expected. |
| `02_synthetic_copy_move.jpg` | A 200×160 block copied inside the image. Copy-move finds a consistent cluster, while metadata looks untouched (metadata alone misses edits). |
| `03_synthetic_spliced.jpg` | Foreign block pasted into a background previously saved at quality 75, then saved at 90. **Run ELA at quality 75** to see it; at 90 it is barely visible, which is itself a teaching point. A simulated "GIMP" Software tag and a modified timestamp trigger the two metadata indicators. |
| `04_synthetic_no_metadata.png` | PNG without EXIF. Missing metadata is a neutral observation, and ELA runs only as a labelled experiment. |
| `05_repetitive_pattern_unedited.jpg` | **Unedited** tiled pattern that copy-move matches anyway: an honest false-positive example. |
| `06_corrupt_truncated.jpg`, `07_not_an_image.png` | Error handling: both are rejected with clear messages. |

All demo images are **synthetic**. They do not represent real-world forensic accuracy.

**Suggested 5-minute demo**
1. Upload `01` and walk through intake (hash computed first), metadata and the integrity page. Copy the
   hash, paste it as the reference and click Verify to get MATCH. Change one character to get MISMATCH.
2. Upload `02` and click **Run all checks**. On Copy-Move, show the coloured cluster joining the two regions.
3. Upload `03`, open ELA, set the quality to 75 and run it. Show the outlier tiles, then the quality sweep.
   Show the metadata indicators.
4. Upload `05` and show that copy-move flags an *unedited* image. Explain why indicators are not proof.
5. Upload `06`/`07` to show graceful rejection.
6. On Results & Report, show the three-state summary, then generate and open the PDF.

## 9. Screenshot checklist (for the project report)

- [ ] Evidence Intake: empty state with the 5-step guide
- [ ] Evidence Intake: preview, file information table and SHA-256
- [ ] Rejected file error (`06_corrupt_truncated.jpg`)
- [ ] Metadata: key fields, indicators and observations (`03`)
- [ ] Integrity: MATCH and MISMATCH callouts
- [ ] ELA: original vs amplified, heat map, outlier tiles at Q = 75 (`03`), quality sweep grid
- [ ] Copy-move: cluster visualisation and region table (`02`), false positive (`05`)
- [ ] Consistency tools: noise map, edges, channels
- [ ] Results & Report: summary callout, check status, visual evidence
- [ ] PDF report: first page and findings/limitations pages
- [ ] Terminal: `pytest` run showing all tests passing

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` not recognised | Activate the venv first, or run `python -m streamlit run app.py`. |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry. |
| `ImportError: DLL load failed` / `libGL.so.1` for cv2 | Make sure **only** `opencv-python-headless` is installed in the venv (`pip uninstall opencv-python opencv-contrib-python`). |
| Upload rejected as "exceeds 20 MB" | Change `DEFAULT_MAX_FILE_BYTES` in `forensic/validation.py` **and** `maxUploadSize` in `.streamlit/config.toml`. |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| Copy-move is slow on very large images | Lower "Max ORB keypoints" or keep the default analysis size (1,600 px). |
| ELA looks "bright everywhere" | Expected for PNGs, screenshots and detailed textures. Read the limitations and try the quality sweep. |
| `ModuleNotFoundError: forensic` in tests | Run `pytest` from the project root. `pytest.ini` adds the root to `pythonpath`. |

## 11. Limitations (summary)

* No technique here can prove authenticity or manipulation. All outputs are indicators for a human.
* No calibrated model is used, so there are no probabilities or confidence scores.
* Metadata is easily stripped or forged.
* ELA depends on compression history, content and the chosen quality.
* Copy-move misses smooth-region and mirrored copies and false-alarms on repetitive patterns.
* Splicing detection relies only on ELA and visual aids. There is no learned detector.
* No formal chain of custody: no signed audit log, access control or trusted timestamping.
  Results are not legally admissible.

## 12. Future scope

* Dense block-based copy-move methods (Zernike/PCT moments, PatchMatch) for smooth regions, plus mirror
  invariance.
* Full JPEG ghost analysis and double-quantisation (DCT histogram) detection.
* Camera sensor noise (PRNU) fingerprinting when reference images from the camera are available.
* An optional PyTorch splicing/localisation model (such as a ManTra-Net or TruFor-style network), only
  if it is evaluated and calibrated on a public benchmark before any score is shown.
* Signed, append-only audit logs and trusted timestamps toward a chain-of-custody workflow.
* Batch analysis, HEIC/WebP/TIFF support and C2PA content-credential verification.

## 13. References (official documentation and key papers)

* Streamlit docs: https://docs.streamlit.io/
* OpenCV ORB feature tutorial: https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
* OpenCV feature matching: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
* OpenCV `estimateAffinePartial2D`: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
* Pillow docs (Image, ExifTags): https://pillow.readthedocs.io/en/stable/
* Python `hashlib`: https://docs.python.org/3/library/hashlib.html
* ReportLab user guide: https://docs.reportlab.com/
* NumPy: https://numpy.org/doc/stable/
* pytest: https://docs.pytest.org/
* FIPS 180-4, Secure Hash Standard (SHA-256), NIST.
* N. Krawetz, "A Picture's Worth... Digital Image Analysis and Forensics", Black Hat 2007 (ELA).
* H. Farid, "Exposing Digital Forgeries from JPEG Ghosts", IEEE TIFS 2009.
* I. Amerini et al., "A SIFT-Based Forensic Method for Copy–Move Attack Detection and Transformation
  Recovery", IEEE TIFS 2011 (g2NN matching and clustering).
* E. Rublee et al., "ORB: An efficient alternative to SIFT or SURF", ICCV 2011.
* M. Fischler and R. Bolles, "Random Sample Consensus", CACM 1981.

## 14. Disclaimer

ImageSleuth AI is an educational project. Its outputs are indicators that need expert interpretation.
It does not determine authenticity, is not validated on forensic benchmarks, does not provide a chain of
custody, and must not be presented as certified or legally admissible forensic evidence.
