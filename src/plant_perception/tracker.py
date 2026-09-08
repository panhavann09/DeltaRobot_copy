#!/usr/bin/env python3
"""
tracker.py — Object tracking: ByteTrack (primary) with FallbackIoU (pure NumPy).
Provides robust tracking ID assignment, direct-to-fallback 1-to-1 association, and error reporting.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    HAS_BYTETRACK = True
except Exception:
    BYTETracker = None
    HAS_BYTETRACK = False

CROP_CLASSES = {0, 1}
WEED_CLASSES = {2, 3}


class DetectionResults:
    """Adapter wrapping detections for the Ultralytics BYTETracker API."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        return DetectionResults(self.xywh[index], self.conf[index], self.cls[index])


class TrackerArgs:
    """Configuration holder for ByteTrack."""

    def __init__(self, track_high_thresh=0.35, track_low_thresh=0.1,
                 new_track_thresh=0.40, track_buffer=15, match_thresh=0.75,
                 fuse_score=True):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score


def _create_bytetracker(args, frame_rate=15):
    """Instantiate BYTETracker safely across Ultralytics API versions with frame_rate support."""
    if BYTETracker is None:
        return None, False

    import inspect
    # Strategy 1: Check signature via inspect
    try:
        sig = inspect.signature(BYTETracker.__init__)
        if "frame_rate" in sig.parameters:
            return BYTETracker(args, frame_rate=frame_rate), True
    except Exception:
        pass

    # Strategy 2: Attempt keyword initialization
    try:
        return BYTETracker(args, frame_rate=frame_rate), True
    except TypeError:
        pass

    # Strategy 3: Attempt positional initialization
    try:
        return BYTETracker(args, frame_rate), True
    except TypeError:
        pass

    # Strategy 4: Fallback single argument initialization
    try:
        return BYTETracker(args), False
    except Exception as exc:
        logger.warning("Failed to initialize BYTETracker: %s", exc)
        return None, False


class ByteTrackWrapper:
    """Wraps Ultralytics BYTETracker and maps IDs back to Detection objects."""

    ID_MAP_IOU_THRESH = 0.3
    _warned_frame_rate_unsupported = False

    def __init__(self, track_high_thresh=0.35, track_low_thresh=0.1,
                 new_track_thresh=0.40, track_buffer=15, match_thresh=0.75,
                 frame_rate=15):
        self.args = TrackerArgs(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )
        self.frame_rate = frame_rate
        self.tracker, frame_rate_supported = _create_bytetracker(self.args, frame_rate=frame_rate)

        if not frame_rate_supported and BYTETracker is not None:
            if not ByteTrackWrapper._warned_frame_rate_unsupported:
                logger.warning("[ByteTrackWrapper] BYTETracker constructor does not accept frame_rate parameter; falling back to single-argument constructor.")
                ByteTrackWrapper._warned_frame_rate_unsupported = True

        self._warned_fallback = False
        self._warned_empty_update_failure = False

    def reset(self):
        if self.tracker is not None:
            try:
                self.tracker.reset()
            except Exception:
                pass
        self._warned_fallback = False
        self._warned_empty_update_failure = False

    def update(self, detections):
        """Update tracker state and assign track_id to input detections."""
        if not detections:
            if self.tracker is not None:
                try:
                    self.tracker.update(DetectionResults([], [], []))
                except Exception as exc:
                    if not self._warned_empty_update_failure:
                        logger.warning("ByteTrack empty-frame update failed: %s", exc)
                        self._warned_empty_update_failure = True
            return detections

        xywh, conf, cls = [], [], []
        for det in detections:
            x1, y1, x2, y2 = det.box_xyxy
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            xywh.append([x1 + w / 2.0, y1 + h / 2.0, w, h])
            conf.append(det.confidence)
            cls.append(det.class_id)

        det_results = DetectionResults(xywh, conf, cls)

        raw_output = None
        if self.tracker is not None:
            try:
                raw_output = self.tracker.update(det_results)
            except Exception as exc:
                if not self._warned_empty_update_failure:
                    logger.warning(f"[ByteTrackWrapper] BYTETracker.update() exception: {exc}")
                    self._warned_empty_update_failure = True

        active_tracks = [t for t in getattr(self.tracker, "tracked_stracks", []) if getattr(t, "is_activated", True)]
        if not active_tracks and hasattr(self.tracker, "stracks"):
            active_tracks = [t for t in getattr(self.tracker, "stracks", []) if getattr(t, "is_activated", True)]

        self._assign_track_ids(detections, raw_output, active_tracks)
        return detections

    def _parse_bytetrack_results(self, raw_output: np.ndarray, detections: list) -> tuple[dict[int, int], list]:
        """
        Returns:
            direct_mapping: dict[detection_index -> track_id]
            unmatched_track_rows: list of raw output rows/tracks that could not be mapped directly
        """
        direct_mapping = {}
        unmatched_rows = []

        if not isinstance(raw_output, np.ndarray) or raw_output.ndim != 2 or raw_output.shape[1] < 5:
            return direct_mapping, unmatched_rows

        num_cols = raw_output.shape[1]
        used_dets = set()
        used_tids = set()

        # Check if det_idx column (col index 7) is available
        has_det_idx = (num_cols >= 8)

        for row in raw_output:
            t_id_val = row[4]
            if not np.isfinite(t_id_val):
                continue
            t_id = int(round(t_id_val))

            mapped = False
            if has_det_idx:
                d_idx_val = row[7]
                if np.isfinite(d_idx_val) and abs(d_idx_val - round(d_idx_val)) < 1e-3:
                    d_idx = int(round(d_idx_val))
                    if 0 <= d_idx < len(detections) and d_idx not in used_dets and t_id not in used_tids:
                        direct_mapping[d_idx] = t_id
                        used_dets.add(d_idx)
                        used_tids.add(t_id)
                        mapped = True

            if not mapped:
                unmatched_rows.append(row)

        return direct_mapping, unmatched_rows

    def _assign_track_ids(self, detections, raw_output, active_tracks):
        """Map track IDs using direct detection index first, then 1-to-1 fallback association."""
        for det in detections:
            det.track_id = None

        if not detections:
            return

        assigned_detection_indices = set()
        used_track_ids = set()

        # Step 1: Attempt direct mapping from raw 2D numpy output array
        direct_mapping, unmatched_rows = self._parse_bytetrack_results(raw_output, detections)
        for d_idx, t_id in direct_mapping.items():
            detections[d_idx].track_id = t_id
            assigned_detection_indices.add(d_idx)
            used_track_ids.add(t_id)

        # Step 2: Attempt direct mapping from STrack object attributes if raw array gave partial results
        if active_tracks:
            for track in active_tracks:
                t_id = getattr(track, "track_id", None)
                if t_id is None or t_id in used_track_ids:
                    continue
                d_idx = getattr(track, "idx", getattr(track, "ind", getattr(track, "det_idx", None)))
                if d_idx is not None and isinstance(d_idx, (int, np.integer)):
                    d_idx = int(d_idx)
                    if 0 <= d_idx < len(detections) and d_idx not in assigned_detection_indices:
                        detections[d_idx].track_id = t_id
                        assigned_detection_indices.add(d_idx)
                        used_track_ids.add(t_id)

        # Step 3: Run 1-to-1 fallback association for unassigned detections and unused tracks
        unassigned_dets = [i for i in range(len(detections)) if i not in assigned_detection_indices]
        if unassigned_dets:
            if not self._warned_fallback:
                logger.warning("[ByteTrackWrapper] Direct detection index unavailable for all rows. Running 1-to-1 class-aware IoU fallback association.")
                self._warned_fallback = True

            pairs = []
            if isinstance(raw_output, np.ndarray) and raw_output.ndim == 2 and raw_output.shape[1] >= 5:
                for d_idx in unassigned_dets:
                    det = detections[d_idx]
                    det_fam = 0 if det.class_id in CROP_CLASSES else 2
                    for row in raw_output:
                        t_id = int(round(row[4]))
                        if t_id in used_track_ids:
                            continue
                        t_cls = int(round(row[6])) if raw_output.shape[1] >= 7 else None
                        if t_cls is not None:
                            t_fam = 0 if t_cls in CROP_CLASSES else 2
                            if t_fam != det_fam:
                                continue

                        t_xyxy = row[:4]
                        iou = _iou(det.box_xyxy, t_xyxy)
                        if iou >= self.ID_MAP_IOU_THRESH:
                            pairs.append((iou, d_idx, t_id))
            else:
                for d_idx in unassigned_dets:
                    det = detections[d_idx]
                    det_fam = 0 if det.class_id in CROP_CLASSES else 2
                    for track in active_tracks:
                        t_id = getattr(track, "track_id", None)
                        if t_id is None or t_id in used_track_ids:
                            continue
                        t_cls = getattr(track, "cls", getattr(track, "class_id", None))
                        if t_cls is not None:
                            t_fam = 0 if int(t_cls) in CROP_CLASSES else 2
                            if t_fam != det_fam:
                                continue

                        t_xyxy = getattr(track, "xyxy", getattr(track, "tlbr", None))
                        if t_xyxy is None:
                            continue
                        iou = _iou(det.box_xyxy, t_xyxy)
                        if iou >= self.ID_MAP_IOU_THRESH:
                            pairs.append((iou, d_idx, t_id))

            pairs.sort(key=lambda x: x[0], reverse=True)
            matched_dets, matched_tids = set(), set()
            for iou_val, d_idx, t_id in pairs:
                if d_idx in matched_dets or t_id in matched_tids:
                    continue
                detections[d_idx].track_id = t_id
                matched_dets.add(d_idx)
                matched_tids.add(t_id)


class _FallbackTrack:
    def __init__(self, track_id, detection):
        self.track_id = track_id
        self.box_xyxy = np.copy(detection.box_xyxy)
        self.class_id = detection.class_id
        self.confidence = detection.confidence
        self.lost_frames = 0

    def update(self, detection):
        self.box_xyxy = np.copy(detection.box_xyxy)
        self.class_id = detection.class_id
        self.confidence = detection.confidence
        self.lost_frames = 0


class FallbackIoUTracker:
    """Greedy IoU-based tracker requiring only NumPy."""

    def __init__(self, track_buffer=15, match_thresh=0.3):
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.next_id = 1
        self.tracks = []

    def reset(self):
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        if not detections:
            for t in self.tracks:
                t.lost_frames += 1
            self.tracks = [t for t in self.tracks if t.lost_frames <= self.track_buffer]
            return detections

        matches = []
        for d_idx, det in enumerate(detections):
            det_fam = 0 if det.class_id in CROP_CLASSES else 2
            for t_idx, track in enumerate(self.tracks):
                trk_fam = 0 if track.class_id in CROP_CLASSES else 2
                if det_fam != trk_fam:
                    continue
                iou = _iou(det.box_xyxy, track.box_xyxy)
                if iou >= self.match_thresh:
                    matches.append((iou, d_idx, t_idx))

        matched_dets, matched_tracks = set(), set()
        matches.sort(key=lambda x: x[0], reverse=True)
        for iou_val, d_idx, t_idx in matches:
            if d_idx in matched_dets or t_idx in matched_tracks:
                continue
            self.tracks[t_idx].update(detections[d_idx])
            detections[d_idx].track_id = self.tracks[t_idx].track_id
            matched_dets.add(d_idx)
            matched_tracks.add(t_idx)

        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.lost_frames += 1

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = _FallbackTrack(self.next_id, det)
                self.next_id += 1
                self.tracks.append(new_track)
                det.track_id = new_track.track_id

        self.tracks = [t for t in self.tracks if t.lost_frames <= self.track_buffer]
        return detections


def _iou(box_a, box_b):
    """IoU between two (x1, y1, x2, y2) boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    union = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """Unified interface — picks ByteTrack or FallbackIoU automatically."""

    def __init__(self, tracker_type="bytetrack", track_high_thresh=0.35,
                 track_low_thresh=0.1, new_track_thresh=0.40, track_buffer=15,
                 match_thresh=0.75, frame_rate=15):
        tracker_type = tracker_type.lower()

        if tracker_type == "bytetrack" and HAS_BYTETRACK:
            self._impl = ByteTrackWrapper(
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                frame_rate=frame_rate,
            )
            logger.info("[Tracker] Using ByteTrack.")
        else:
            if tracker_type == "bytetrack" and not HAS_BYTETRACK:
                logger.info("[Tracker] ByteTrack unavailable — falling back to IoU tracker.")
            iou_thresh = min(0.4, match_thresh)
            self._impl = FallbackIoUTracker(track_buffer=track_buffer, match_thresh=iou_thresh)
            logger.info(f"[Tracker] Using FallbackIoU (thresh={iou_thresh}).")

    def reset(self):
        self._impl.reset()

    def update(self, detections):
        return self._impl.update(detections)
