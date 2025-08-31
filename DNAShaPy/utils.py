import os
import sys
import gzip
import numpy as np
from importlib.resources import files
from pathlib import Path
import requests
import pooch
import tarfile
import shutil

def untar_clean(fname, action, pooch_instance):
    """Pooch processor: unpack a shape-cache tarball into the cache directory.

    Extraction strategy:
    1. The archive is extracted into a temporary staging directory alongside
       the target cache directory (``<cache_dir>/<archive>.__extracting``).
    2. If the archive contains a single top-level directory it is flattened;
       all ``.npy`` files are moved directly into ``cache_dir``.
    3. Only ``.npy`` payload files are retained (any other files are ignored).
    4. The staging directory and original ``.tar.gz`` are removed on success.

    Parameters
    ----------
    fname : str or Path
        Path to the downloaded tar.gz file provided by pooch.
    action : str
        Unused parameter required by the pooch processor signature.
    pooch_instance : pooch.Pooch
        The calling pooch instance (unused here, but part of the contract).

    Returns
    -------
    list[str]
        List of absolute paths to the extracted ``.npy`` files (what pooch
        expects a processor to return).
    """

    fname = Path(fname)
    cache_dir = fname.parent  
    staging = cache_dir / (fname.stem + "__extracting")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fname, "r:gz") as tar:

        tar.extractall(path=staging, filter="data")


    entries = list(staging.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        top = entries[0]
        sources = list(top.rglob("*"))
    else:
        sources = list(staging.rglob("*"))

    moved_paths = []
    for src in sources:
        if not src.is_file():
            continue
            
        if src.suffix != ".npy":
            continue
        dest = cache_dir / src.name
        shutil.move(str(src), dest)
        moved_paths.append(dest)

    # Cleanup staging directory and original archive
    shutil.rmtree(staging, ignore_errors=True)
    try:
        fname.unlink()
    except FileNotFoundError:
        pass


    # Return list of unpacked file paths (what pooch expects)
    return [str(p) for p in moved_paths]

def make_layer_marker(dirpath, archive_hash):
    """Return a Path to the marker file for a given archive hash.

    The presence of this file signals a successful prior download & extract
    for the matching hash, allowing ``get_cache`` to skip re-downloading.

    Parameters
    ----------
    dirpath : Path
        Cache directory.
    archive_hash : str
        Hash string as used in the pooch registry (``algo:hexdigest``).

    Returns
    -------
    Path
        Path object for the marker file.
    """

    return dirpath / f".complete-{archive_hash.replace(':','-')}"

def get_cache(cache_path=None, layer=4):
    """Ensure the on-disk shape cache for a given layer is present.

    If the layer's marker file exists, no action is taken. Otherwise the
    corresponding layer archive is fetched (via pooch) and extracted into
    ``cache_path``. For layer 5, the CLI coordinates fetching "part_1" and
    "part_2" separately by calling this function twice with ``layer`` values
    ``"5_part_1"`` and ``"5_part_2"``.

    Parameters
    ----------
    cache_path : str | Path | None
        Destination directory. If ``None`` a package data path is attempted
        and falls back to the platform cache (``pooch.os_cache('DNAShaPy')``).
    layer : int | str
        Layer identifier (0–4 as int, or "5_part_1" / "5_part_2" for layer 5
        split archives).

    Raises
    ------
    ValueError
        If no registered checksum exists for the constructed archive name.
    SystemExit
        On unrecoverable HTTP errors when downloading.
    """

    DEFAULT_BASE_URL = f"https://github.com/harmstonlab/DNAShaPy/releases/download/cache_archives/"

    ARCHIVE_NAME = f"shape_cache_layer_{layer}.tar.gz"
    ARCHIVE_HASHES = {"shape_cache_layer_0.tar.gz":"sha256:0d42b7a27d2140db65d3d2832144ce6c70ba7126f17b2583eb4f3f01e5ce74fc",
                      "shape_cache_layer_1.tar.gz":"sha256:043ddf6dadf923882d0f11a214ae786b5827829b3f77fd14edbeb0c3875914df",
                      "shape_cache_layer_2.tar.gz":"sha256:9ac781173b78daaaa330210a6af3a171a6650abc5bf55d244891bcfffdcc6825",
                      "shape_cache_layer_3.tar.gz":"sha256:b2b680fb1f5f2ba69649bb1b8d5eef4c73d7f3f21420a5637706d8717a822d86",
                      "shape_cache_layer_4.tar.gz":"sha256:e27d353014b1985d3d52a5fa6f9b31896d513291a49f72657bac81dd94aaab2b",
                      "shape_cache_layer_5_part_1.tar.gz":"sha256:70a386f79e7620118026ce74a92b65a1e5979399a3f617a445984701cff0236d",
                      "shape_cache_layer_5_part_2.tar.gz":"sha256:b8dfe57b7a9ed6323efb6f2567e55eca5ac67fb8fd93a61261458cc65bbcb0a7",
    }
    if ARCHIVE_NAME not in ARCHIVE_HASHES:
        raise ValueError(f"No checksum registered for {ARCHIVE_NAME}")
    
    
    REGISTRY = {ARCHIVE_NAME: ARCHIVE_HASHES[ARCHIVE_NAME]}
    if cache_path is None:
        try:
            cache_path = files("DNAShaPy").joinpath("shape_cache")
        except:
            cache_path = pooch.os_cache("DNAShaPy")
    marker = make_layer_marker(cache_path, ARCHIVE_HASHES[ARCHIVE_NAME])
    if marker.exists():
        print(f"Hash marker exists for layer {layer}")
        print("Skipping Download")
        print(f"If cache has become corrupted or is missing files remove {marker} and retry")
        return 
    try:
        CACHE = pooch.create(
            path=cache_path,
            base_url=DEFAULT_BASE_URL,
            registry=REGISTRY,
        )
        CACHE.fetch(
            ARCHIVE_NAME,
            processor=untar_clean,
        )
        marker.write_text("ok\n")
    except requests.exceptions.HTTPError as e:
        print(f"Error downloading {ARCHIVE_NAME}: {e}")
        sys.exit(1)

def format_vals(vals, sep):
    """Format a 1D float array to a single delimited string.

    Values formatted; NaNs are emitted as
    literal ``"NA"``.

    Parameters
    ----------
    vals : array-like of float
    Predicted numeric values.
    sep : str
    Separator to join values with (e.g., "\t" or " ").

    Returns
    -------
    str
    Joined string suitable for writing to text outputs.
    """

    s = np.char.mod("%g", vals)
    if np.isnan(vals).any():
        s = np.where(np.isnan(vals), "NA", s)
    return sep.join(s.tolist())


def check_equal_length(records,quiet):
    """Verify all FASTA records have identical length.

    This is intended for enforcing ``--means-only`` preconditions. Accepts a
    file path or a file-like handle already opened in text mode. The stream
    position is rewound to the beginning on return.

    Parameters
    ----------
    records : list of tuples 
    (id, seq)

    Raises
    ------
    ValueError
    If any record length differs from the first sequence.
    """

    expected_len = None
    if not quiet:
        print("Checking that all sequences have equal length", file=sys.stdout)
    for id, seq in records:
        if expected_len is None:
            expected_len = len(seq)
        elif len(seq) != expected_len:
            raise ValueError(
                f"Sequence {id} has length {len(seq)}, expected {expected_len}\nFor --means-only, all sequences must be of equal length."
            )
    if not quiet:
        print(f"All {len(records)} sequences have length {expected_len}", file=sys.stdout)


def open_maybe_gz(path, mode):
    """Open a path that may be gzip-compressed or ``-`` for stdio.

    Parameters
    ----------
    path : str
    File path or ``"-"`` for stdin/stdout.
    mode : str
    Standard file mode (e.g., ``"rt"``, ``"wt"``).

    Returns
    -------
    IOBase
    Opened handle (may be stdin/stdout if ``path == '-'``).
    """

    if path == "-":
        return sys.stdin if "r" in mode else sys.stdout
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def count_fasta_records(handle):
    """Count the number of FASTA records by scanning header lines.

    Parameters
    ----------
    handle : IOBase
    Text-mode file handle positioned at the beginning of a FASTA file.

    Returns
    -------
    int
    Number of lines starting with ``">"``. The handle is rewound on exit.
    """

    try:
        return sum(1 for line in handle if line.startswith(">"))
    finally:
        if handle not in (sys.stdin, sys.stdout):
            handle.seek(0)


def is_dir_like(p):
    """Return True if ``p`` looks like a directory destination.

    A path is treated as directory-like if it ends with the OS separator or
    if it exists and is an actual directory.

    Parameters
    ----------
    p : str
    Filesystem path.

    Returns
    -------
    bool
    Whether the path is directory-like.
    """

    return p.endswith(os.sep) or (os.path.exists(p) and os.path.isdir(p))

def build_cache(predictor, features, layer):
    """Build np_array cache files for a set of features at a given layer.

    Parameters
    ----------
    predictor : Predictor
    Initialized Predictor instance.
    features : set of str
    Set of feature names to build.
    layer : int
    Padding radius.

    Returns
    -------
    tuple of (int, int)
    Number of files built and number of files skipped.
    """

    built, skipped = 0, 0
    for f in features:
        k = predictor._k_for(f, layer)
        
        if os.path.exists(predictor._array_path(f, k)):
            if not predictor.quiet:
                print(f"[cache] skipping existing {f} k={k}", file=sys.stdout)
            skipped += 1
        else:
            predictor._np_array_table(f, k)
            built += 1
    return (built, skipped)