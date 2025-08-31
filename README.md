# DNAShaPy

DNAShaPy is a Python package for predicting DNA shape features from nucleotide sequences using pre-computed numpy arrays derived from DeepDNAShape. It provides fast predictions for a variety of inter- and intra-base features, supporting both command line and programmatic use.

## Installation

You can install DNAShaPy using `pip`:

```sh
git clone git@github.com:harmstonlab/DNAShaPy.git
cd DNAShaPy
pip install .
```

Or install directly without cloning:
```sh
pip install git+git@github.com:harmstonlab/DNAShaPy.git
```

**Dependencies:**  
DNAShaPy requires Python 3.8+ and the following packages (automatically installed with pip):

- biopython
- fastparquet
- numba
- numpy
- pandas
- pyarrow
- tqdm
- pooch

## First use
Before running predictions the pre-computed lookup tables (``.npy`` shape cache
files) must either be downloaded or generated locally.

### Download numpy arrays (RECOMMENDED)

To download the files you can run:
```sh
DNAShaPy --get-cache --cache-dir /path/to/where/you/want/cache/stored --layer 4
```
- `--cache-dir` (optional): directory to store cache files. (NOTE - `--layer 5` cache is split into two parts due to the large size. However using `--layer 5` will download both parts)
    - If omitted, DNAShaPy will try to use a default cache directory (see [Inputs](#inputs)).
- `--layer` (optional): which cache layer to download.
    - The cache is split into one tarball per layer to keep downloads smaller.
    - Default is `--layer 4`.
    - You can run the command multiple times with different `--layer` values to populate the same cache directory with multiple layers.

- Marker file: for each layer, a hidden marker file is written to the cache directory after successful download and extraction.
    - This prevents re-downloading the same layer.
    - If a cache becomes corrupted or incomplete, simply delete the corresponding marker file and re-run the command.

The cache tarballs can also be downloaded manually from the project GitHub Releases page.

### Download parquet tables (alternative)

Alternatively pre-computed parquet files are available from the
[Deep DNAShape Webserver Publication](https://doi.org/10.1093/nar/gkae433)
([direct link to files](https://figshare.com/articles/dataset/Query_tables_in_Parquet_format_for_Deep_DNAshape_webserver/25286197)).

You can then run either:
```sh
DNAShaPy --build-cache (--feature {FEATURE} | --all) (--layer {LAYER}) --cache-dir /path/to/cache
```
Again, the `--cache-dir` parameter is optional. And `--layer` is the number of surrounding nucleotides to consider (see [Inputs](#Inputs)). This will pre-build the `npy` array files to be used in future runs for the given feature(s).

Alternatively you can just run `DNAShaPy` with the parquet files present and it
will build the cache and save it for future runs. NOTE: if running many
instances concurrently (e.g. on an HPC cluster) and the cache is not
pre-created, each instance will attempt to build the same arrays leading to
unnecessary memory usage. Pre-computing or downloading the cache first is
highly recommended.

## Inputs
Required inputs:
- `--input` Path to the FASTA file to analyse. Can be compressed (`.gz`). Alternatively this can be set to `-` to read from `stdin`.
- `--feature` Either a single feature or a comma-separated list of features to predict (run `DNAShaPy --list-features` to list all available features). Mutually exclusive with `--all`.
- `--all` Predicts shape values to all features (run `DNAShaPy --list-features` to be shown a list of all available features)

Optional inputs:
- `--layer` A value between 1-4 for inter-base (`bpstep`) features and 1-5 for intra-base (`bp`) features. This is the number of flanking bases to consider (e.g. for an intra-base feature (e.g. Stretch) a layer value of 4 will result in the prediction for a given base being made with the context of the 4 flanking bases either side, resulting in a 9bp sequence being used for the prediction). For more information about `layer` please refer to the original [Deep DNAShape Paper](https://doi.org/10.1038/s41467-024-45191-5) and [Deep DNAShape Webserver Paper](https://doi.org/10.1093/nar/gkae433)
- `--output` File(s) to store output. Defaults to `stdout`; if more than one feature (or `--all`) is used then `stdout` cannot be used and a file must be provided. This file must include `{feature}` which will be replaced with the predicted feature, or a directory can be given and files will be placed there named `{feature}.txt`. Output files can also be `.gz`.
- `--cache-dir` Location to read `.npy` cache files. Cache files will also be placed here when they are computed/retrieved. By default this is `{INSTALLATION_PATH}/shape_cache`
- `--bp` Path to the `bp.parquet` file. This is only needed if you are computing `npy` array files yourself
- `--bpstep` Path to the `bpstep.parquet` file. This is only needed if you are computing `npy` array files yourself
- `--headers` If this flag is present output values will be in `FASTA` format, including input sequence names
- `--means-only` If present, and more than one sequence is in the FASTA file, output the mean value at each position rather than one value per sequence. NOTE: Requires all sequences to be the same length (validated beforehand).
- `--quiet` If this flag is present minimal information is printed. This removes cache messages and progress bars. If `--output` is `stdout` then `--quiet` is forced

## Outputs

DNAShaPy outputs per-position predictions for each requested feature:

- Space-delimited text files with predicted values (one row per sequence).
- Optionally, mean values across all positions (for equal-length sequences).
- NaN is used for positions where prediction is not possible (this is a fail safe and should not occur as parquet / npy files should contain all possible N-mers).

## Usage Example

**Command-line:**

```sh
DNAShaPy --input sequences.fasta --feature MGW,ProT --output out/{feature}.txt
```

**Python API:**
```python
from DNAShaPy.Predictor import Predictor

# Locate cache directory
from importlib.resources import files
cache_dir=files("DNAShaPy").joinpath("shape_cache")

predictor = Predictor(cache_dir=cache_dir)
seq = "ACGTACGTACGT"
preds = predictor.predict_seq(seq, feature="MGW")
print(preds)
```
## References
This work is built on top of, and relies upon, both the original [Deep DNAShape Paper](https://doi.org/10.1038/s41467-024-45191-5) and files provided as part of the [Deep DNAShape Webserver Paper](https://doi.org/10.1093/nar/gkae433)


