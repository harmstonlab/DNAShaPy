import os
import sys

import numba as nb
import numpy as np
import pandas as pd


class Predictor:
    """Predictor for bp/bp-step DNA shape features.

    Lazily loads parquet tables for bp and bp-step feature sets, exposes a
    ``predict_seq`` method to compute per-position predictions using np_array
    k-mer lookup tables that can be cached to disk.

    Parameters
    ----------
    bp_path : str
    Path to the base-pair (bp) parquet table.
    bpstep_path : str
    Path to the base-pair-step (bp-step) parquet table.
    quiet : bool, default False
    Suppress informational prints (progress/logging).
    cache_dir : str or None, default None
    Directory where np_array ``.npy`` arrays are stored/loaded. If set,
    the directory is created on initialization.
    """

    def __init__(
        self,
        bp_path=None,
        bpstep_path=None,
        quiet=True,
        cache_dir=None,
    ):
        self.bpstep_features = {
            "Shift",
            "Slide",
            "Rise",
            "Tilt",
            "Roll",
            "HelT",
            "Shift-FL",
            "Slide-FL",
            "Rise-FL",
            "Tilt-FL",
            "Roll-FL",
            "HelT-FL",
        }
        self.bp_features = {
            "MGW",
            "EP",
            "Opening",
            "ProT",
            "Buckle",
            "Stagger",
            "Stretch",
            "Shear",
            "MGW-FL",
            "Opening-FL",
            "ProT-FL",
            "Buckle-FL",
            "Stagger-FL",
            "Stretch-FL",
            "Shear-FL",
        }
        self.allfeatures = self.bp_features | self.bpstep_features

        self._bp_path = bp_path
        self._bpstep_path = bpstep_path
        self.bp_data = None
        self.bpstep_data = None

        self._series_cache = {}
        self._np_array_cache = {}
        self.cache_dir = cache_dir
        self.quiet = quiet
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _load_with_kmer_index(self, parquet_path, quiet=False):
        """Load a parquet table and index it by a k-mer column.

        Parameters
        ----------
        parquet_path : str
        Path to the parquet file.
        quiet : bool, default False
        If False, prints a short "Loaded" message to stdout.


        Returns
        -------
        pandas.DataFrame
        DataFrame indexed by uppercase k-mer strings.
        """

        df = pd.read_parquet(parquet_path)
        idx = df.index.astype(str).str.upper()
        if len(idx) and not idx.map(lambda s: set(s) <= set("ACGTN")).all():
            raise ValueError(
                f"{parquet_path}: index/column used as k-mer contains non-ACGTN characters."
            )
        if not quiet:
            print(f"Loaded parquet table: {parquet_path}", file=sys.stdout)
        return df

    def _k_for(self, feature, layer):
        """Return k-mer length for a feature at a given layer.

        For bp-step features, ``k = 2*layer + 2`` and ``layer`` must be ``< 5``.
        For bp features, ``k = 2*layer + 1``.

        Parameters
        ----------
        feature : str
        Name of the feature.
        layer : int
        Padding radius (as passed by CLI).

        Returns
        -------
        int or None
        k-mer length, or ``None`` if the request is invalid (e.g.,
        bp-step with ``layer >= 5``), in which case the caller emits an
        empty result.
        """

        if feature in self.bpstep_features:
            if layer >= 5:
                return None
            return layer * 2 + 2
        return layer * 2 + 1

    def _array_path(self, feature, k):
        """Compute on-disk path for a ``.npy`` array.

        Parameters
        ----------
        feature : str
        Feature name.
        k : int
        k-mer length.

        Returns
        -------
        str
        Full filesystem path inside ``self.cache_dir``.
        """

        fname = f"{feature}_k{k}.npy"
        return os.path.join(self.cache_dir, fname)

    def _series(self, feature):
        """Return the feature series (pandas.Series) for a given feature.

        The series is cached on first use. Non-numeric series are coerced to
        numeric with ``errors='coerce'``.

        Parameters
        ----------
        feature : str
        Feature name present as a column in either bp or bp-step table.


        Returns
        -------
        pandas.Series
        Series indexed by k-mer strings.

        Raises
        ------
        ValueError
            If the feature is not found in either table.
        """

        s = self._series_cache.get(feature)
        if s is not None:
            return s
        if feature in self.bp_features:
            if self.bp_data is None:
                self.bp_data = self._load_with_kmer_index(self._bp_path, self.quiet)
            df = self.bp_data
        elif feature in self.bpstep_features:
            if self.bpstep_data is None:
                self.bpstep_data = self._load_with_kmer_index(
                    self._bpstep_path, self.quiet
                )
            df = self.bpstep_data
        else:
            raise ValueError(f"Unknown feature: {feature}")
        s = df[feature]
        if not np.issubdtype(s.dtype, np.number):
            s = pd.to_numeric(s, errors="coerce")
        self._series_cache[feature] = s
        return s

    def _np_array_from_series(self, s, k):
        """Create a np_array length-``5**k`` array from a k-mer series.

        Non-matching k-mer lengths are skipped. Entries without a value or
        with invalid characters remain ``NaN``.

        Parameters
        ----------
        s : pandas.Series
        Series indexed by k-mer strings with numeric values.
        k : int
        Target k-mer length.

        Returns
        -------
        np.ndarray[np.float64]
        np_array lookup array ready for fast prediction.
        """

        size = 5**k
        arr = np.full(size, np.nan, dtype=np.float64)
        for kmer, v in s.dropna().items():
            sk = str(kmer).upper()
            if len(sk) != k:
                continue
            code = _code_from_kmer(sk)
            if code >= 0:
                arr[code] = float(v)
        return arr

    def _load_np_array(self, feature, k):
        """Load a cached np_array ``.npy`` file if present and valid.

        Parameters
        ----------
        feature : str
        Feature name.
        k : int
        k-mer length.

        Returns
        -------
        np.ndarray or None
        Numpy array if found with the correct length, else ``None``.
        """

        path = self._array_path(feature, k)
        if not path or not os.path.exists(path):
            return None
        arr = np.load(path)
        if arr.shape[0] != 5**k:
            return None
        if not self.quiet:
            print(f"[cache] loaded {os.path.basename(path)}", file=sys.stdout)
        return arr

    def _save_np_array(self, feature, k, arr, s):
        """Persist a np_array array to disk (no-op if ``cache_dir`` is None).

        Parameters
        ----------
        feature : str
        Feature name.
        k : int
        k-mer length.
        arr : np.ndarray
        np_array lookup array to save.
        s : pandas.Series
        Original feature series (unused; kept for parity with callers).
        """

        path = self._array_path(feature, k)
        if not path:
            return

        np.save(path, arr)
        if not self.quiet:
            print(f"[cache] saved {os.path.basename(path)}", file=sys.stdout)

    def _np_array_table(self, feature, k):
        """Return a np_array lookup array for ``(feature, k)``, building if needed.

        Uses an in-memory cache first, then tries to load from disk. If not
        present, constructs from the feature series and saves to disk when
        ``cache_dir`` is set.

        Parameters
        ----------
        feature : str
        Feature name.
        k : int
        k-mer length.

        Returns
        -------
        np.ndarray[np.float64]
        np_array lookup table of length ``5**k``.
        """

        key = (feature, k)
        if key in self._np_array_cache:
            return self._np_array_cache[key]
        arr = self._load_np_array(feature, k)
        if arr is None:
            if not self.quiet:
                print(
                    f"[cache] building np_array table for {feature} k={k}",
                    file=sys.stdout,
                )
            s = self._series(feature)
            arr = self._np_array_from_series(s, k)
            self._save_np_array(feature, k, arr, s)
        self._np_array_cache[key] = arr
        return arr

    def predict_seq(self, seq, feature, layer=4):
        """Predict values for a single sequence and feature at a given layer.

        Pads the sequence with ``'N' * layer`` on both sides, determines the
        appropriate k-mer length, and returns per-position predictions using
        the np_array lookup table.

        Parameters
        ----------
        seq : str or Bio.Seq.Seq
        Nucleotide sequence (A/C/G/T/N). Case-insensitive.
        feature : str
        Feature name (e.g., "MGW", "ProT", "Shift", ...).
        layer : int
        Padding radius; for bp-step features must be ``<= 4``.

        Returns
        -------
        np.ndarray[np.float64]
        1D predictions array; may be empty if ``k`` is invalid.
        """

        k = self._k_for(feature, layer)
        if k is None:
            return np.empty(0, dtype=np.float64)

        s = ("N" * layer) + str(seq) + ("N" * layer)
        n = len(s)
        L = n - k + 1
        if L <= 0:
            return np.empty(0, dtype=np.float64)

        seq_bytes = np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)
        table = self._np_array_table(feature, k)
        return predict_seq_np_array_b5(seq_bytes, k, table)


_BASE5 = np.full(256, -1, dtype=np.int8)
for ch, val in (
    ("A", 0),
    ("C", 1),
    ("G", 2),
    ("T", 3),
    ("N", 4),
    ("a", 0),
    ("c", 1),
    ("g", 2),
    ("t", 3),
    ("n", 4),
):
    _BASE5[ord(ch)] = val


def _code_from_kmer(kmer):
    """Return base-5 integer code for a k-mer over alphabet A,C,G,T,N.

    Parameters
    ----------
    kmer : str
    K-mer string (case-insensitive). Any character outside A/C/G/T/N
    is considered invalid.

    Returns
    -------
    int
    Non-negative base-5 code if valid; -1 if any invalid character is
    present.
    """

    code = 0
    for ch in kmer:
        b = int(_BASE5[ord(ch)])
        if b < 0:
            return -1
        code = code * 5 + b
    return code


@nb.njit()
def predict_seq_np_array_b5(seq_bytes, k, table):
    """Predict shape values for a sequence using a np_array base-5 Look Up Table.

    This Numba-compiled kernel slides a k-length window across ``seq_bytes``
    (encoded as ASCII bytes), maps each k-mer to a base-5 integer code over
    A/C/G/T/N, and looks up values from ``table``. Positions overlapping an
    invalid base produce ``NaN``.

    Parameters
    ----------
    seq_bytes : np.ndarray[np.uint8]
    ASCII bytes of the sequence (already padded if needed).
    k : int
    K-mer length implied by the selected feature/layer.
    table : np.ndarray[np.float64]
    np_array array of length ``5**k`` containing values per k-mer code.

    Returns
    -------
    np.ndarray[np.float64]
    1D array of length ``max(n-k+1, 0)`` with predictions/NaNs.
    """

    n = seq_bytes.size
    L = n - k + 1 if n >= k else 0
    out = np.empty(L, np.float64)
    if L == 0:
        return out

    # 5^(k-1)
    pow5k_1 = 1
    for _ in range(k - 1):
        pow5k_1 *= 5

    code = 0
    have = 0
    pos = 0

    for i in range(n):
        b = _BASE5[seq_bytes[i]]
        if b < 0:  # invalid char
            have = 0
            code = 0
        else:
            if have < k:
                code = code * 5 + int(b)
                have += 1
            else:
                old_b = _BASE5[seq_bytes[i - k]]
                code = (code - int(old_b) * pow5k_1) * 5 + int(b)

        if have == k:
            out[pos] = table[code]
            pos += 1
        elif i >= k - 1:
            out[pos] = np.nan
            pos += 1

    return out
