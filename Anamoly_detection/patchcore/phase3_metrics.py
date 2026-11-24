"""Bounded-memory pixel and connected-component metrics for native maps."""

import cv2
import numpy as np

BINS = 65_536
FPR_LIMIT = 0.30
DIAGNOSTIC_FPR = 0.01
SIZE_LABELS = ("small_lt64", "medium_64_to_511", "large_ge512")


class RegionTargets:
    def __init__(self, foreground):
        self.foreground = foreground
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        region_sizes = np.digitize(areas, [64, 512])
        self.counts = np.bincount(region_sizes, minlength=len(SIZE_LABELS))
        region_ids = labels[foreground] - 1
        self.pixel_sizes = region_sizes[region_ids]
        self.pixel_weights = 1.0 / areas[region_ids] if count > 1 else np.empty(0, dtype=np.float64)


def normalized_partial_area(fpr, tpr, limit):
    stop = np.searchsorted(fpr, limit, side="right")
    area = np.trapezoid(tpr[:stop], fpr[:stop])
    if fpr[stop - 1] < limit:
        endpoint = np.interp(limit, fpr[stop - 1:stop + 1], tpr[stop - 1:stop + 1])
        area += (limit - fpr[stop - 1]) * (tpr[stop - 1] + endpoint) / 2
    return float(area / limit)


def cumulative(histogram):
    return np.concatenate(([0.0], np.cumsum(histogram[::-1], dtype=np.float64)))


class PixelMetrics:
    def __init__(self, score_max):
        if not np.isfinite(score_max) or score_max <= 0:
            raise ValueError("Invalid common score range")
        self.scale = (BINS - 1) / score_max
        self.positive = np.zeros(BINS, dtype=np.int64)
        self.negative = np.zeros(BINS, dtype=np.int64)
        self.region_overlap = np.zeros((len(SIZE_LABELS), BINS), dtype=np.float64)
        self.region_counts = np.zeros(len(SIZE_LABELS), dtype=np.int64)

    def update(self, score, targets):
        if score.shape != targets.foreground.shape or not np.isfinite(score).all() or score.min() < 0:
            raise ValueError("Invalid score map")
        bins = np.minimum(np.multiply(score, self.scale).astype(np.int32), BINS - 1)
        positive_bins = bins[targets.foreground]
        all_pixels = np.bincount(bins.ravel(), minlength=BINS)
        positive = np.bincount(positive_bins, minlength=BINS)
        self.positive += positive
        self.negative += all_pixels - positive
        for index in range(len(SIZE_LABELS)):
            within_size = targets.pixel_sizes == index
            self.region_overlap[index] += np.bincount(
                positive_bins[within_size], weights=targets.pixel_weights[within_size], minlength=BINS,
            )
        self.region_counts += targets.counts

    def compute(self, coarsen=1):
        positive = cumulative(self.positive.reshape(-1, coarsen).sum(axis=1))
        negative = cumulative(self.negative.reshape(-1, coarsen).sum(axis=1))
        if not positive[-1] or not negative[-1] or not self.region_counts.sum():
            raise ValueError("Both pixel classes and at least one defect are required")
        tpr = positive / positive[-1]
        fpr = negative / negative[-1]
        precision = np.divide(positive, positive + negative, out=np.ones_like(positive), where=(positive + negative) > 0)
        region_curves = self.region_overlap.reshape(len(SIZE_LABELS), -1, coarsen).sum(axis=2)
        result = {
            "pixel_auroc": float(np.trapezoid(tpr, fpr)),
            "pixel_ap": float(np.dot(np.diff(tpr), precision[1:])),
            "aupro": normalized_partial_area(fpr, cumulative(region_curves.sum(axis=0)) / self.region_counts.sum(), FPR_LIMIT),
            "positive_pixels": int(positive[-1]),
            "negative_pixels": int(negative[-1]),
            "size_specific": {},
        }
        for label, count, hist in zip(SIZE_LABELS, self.region_counts, region_curves, strict=True):
            row = {"components": int(count), "aupro": None, "mean_overlap_at_1pct_fpr": None}
            if count:
                pro = cumulative(hist) / count
                row["aupro"] = normalized_partial_area(fpr, pro, FPR_LIMIT)
                row["mean_overlap_at_1pct_fpr"] = float(np.interp(DIAGNOSTIC_FPR, fpr, pro))
            result["size_specific"][label] = row
        return result
