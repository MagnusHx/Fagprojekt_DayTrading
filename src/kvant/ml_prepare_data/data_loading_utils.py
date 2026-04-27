from typing import Any, Dict

import numpy as np


def _print_plain_summary(headers, rows) -> None:
    if not rows:
        print("(empty dataset)")
        return

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def fmt(row):
        return " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def summary(self, display: bool = True) -> Dict[str, Any]:
    """
    Summarize this dataset split (as defined by self.index).

    Computes per-ticker:
      - n: number of samples in this dataset for that ticker
      - y_counts: counts of classes present in the prepared artifact
      - first_ts / last_ts: timestamps at min/max tpos for that ticker

    Also computes an 'overall' summary aggregated across tickers.
    """

    label_ids = tuple(getattr(self.store, "label_ids", (0, 1, 2)))
    bincount_size = (max(label_ids) + 1) if label_ids else 0

    if self.index is None or int(self.index.shape[0]) == 0:
        out = {
            "overall": {
                "n": 0,
                "y_counts": {label: 0 for label in label_ids},
                "first_ts": None,
                "last_ts": None,
            },
            "per_ticker": {},
        }
        if display:
            try:
                from tabulate import tabulate

                headers = ["ticker", "n", *[f"y={label}" for label in label_ids], "first_ts", "last_ts"]
                print(tabulate([], headers=headers))
            except Exception:
                print("(empty dataset)")
        return out

    tids = self.index[:, 0].astype(np.int64, copy=False)
    tposs = self.index[:, 1].astype(np.int64, copy=False)
    uniq_tids = np.unique(tids)

    per_ticker: Dict[str, Any] = {}
    overall_n = 0
    overall_counts = {label: 0 for label in label_ids}
    overall_first_ts = None
    overall_last_ts = None

    for tid in uniq_tids:
        tid_i = int(tid)
        ticker = self.store.tickers_all[tid_i]

        mask = tids == tid
        pos = tposs[mask]
        n = int(pos.shape[0])
        overall_n += n

        y_arr = np.asarray(self.store._labels[tid_i][pos], dtype=np.int64)
        y_arr = y_arr[np.isin(y_arr, label_ids)]
        if len(y_arr):
            counts_arr = np.bincount(y_arr, minlength=bincount_size).astype(np.int64)
        else:
            counts_arr = np.zeros(bincount_size, dtype=np.int64)
        counts = {label: int(counts_arr[label]) for label in label_ids}
        for label in label_ids:
            overall_counts[label] += counts[label]

        ts_arr = self.store._timestamps[tid_i]
        pmin = int(pos.min())
        pmax = int(pos.max())
        first_ts = ts_arr[pmin]
        last_ts = ts_arr[pmax]

        if overall_first_ts is None or first_ts < overall_first_ts:
            overall_first_ts = first_ts
        if overall_last_ts is None or last_ts > overall_last_ts:
            overall_last_ts = last_ts

        per_ticker[ticker] = {
            "tid": tid_i,
            "n": n,
            "y_counts": counts,
            "first_ts": None if first_ts is None else str(np.datetime_as_string(first_ts, unit="s")),
            "last_ts": None if last_ts is None else str(np.datetime_as_string(last_ts, unit="s")),
        }

    out = {
        "overall": {
            "n": int(overall_n),
            "y_counts": {label: int(overall_counts[label]) for label in label_ids},
            "first_ts": None if overall_first_ts is None else str(np.datetime_as_string(overall_first_ts, unit="s")),
            "last_ts": None if overall_last_ts is None else str(np.datetime_as_string(overall_last_ts, unit="s")),
        },
        "per_ticker": per_ticker,
    }

    if display:
        try:
            from tabulate import tabulate

            headers = ["ticker", "n", *[f"y={label}" for label in label_ids], "first_ts", "last_ts"]
            rows = []
            for ticker in sorted(per_ticker.keys()):
                d = per_ticker[ticker]
                rows.append([ticker, d["n"], *[d["y_counts"][label] for label in label_ids], d["first_ts"], d["last_ts"]])

            print(tabulate(rows, headers=headers, tablefmt="github"))

            o = out["overall"]
            print("\nOverall:")
            print(
                tabulate(
                    [[o["n"], *[o["y_counts"][label] for label in label_ids], o["first_ts"], o["last_ts"]]],
                    headers=["n", *[f"y={label}" for label in label_ids], "first_ts", "last_ts"],
                    tablefmt="github",
                )
            )
        except ImportError:
            headers = ["ticker", "n", *[f"y={label}" for label in label_ids], "first_ts", "last_ts"]
            rows = []
            for ticker in sorted(per_ticker.keys()):
                d = per_ticker[ticker]
                rows.append([ticker, d["n"], *[d["y_counts"][label] for label in label_ids], d["first_ts"], d["last_ts"]])
            _print_plain_summary(headers, rows)
            print("\nOverall:")
            o = out["overall"]
            _print_plain_summary(
                ["n", *[f"y={label}" for label in label_ids], "first_ts", "last_ts"],
                [[o["n"], *[o["y_counts"][label] for label in label_ids], o["first_ts"], o["last_ts"]]],
            )
        except Exception as e:
            print("Failed to display summary:", e)

    return out
