#!/usr/bin/env python3
"""
track_state.py — Persistent track management with EMA filtering,
consecutive hit confirmation, semantic family hysteresis stabilization,
and safe current-depth actionability logic.
"""

from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Dict, Tuple
import numpy as np

TENTATIVE = 0
CONFIRMED = 1
COASTING = 2

CROP_CLASSES = {0, 1}
WEED_CLASSES = {2, 3}


@dataclass
class PlantTrackState:
    track_id: int
    class_id: int
    class_name: str
    confidence: float

    bbox_xyxy: np.ndarray
    root_point: np.ndarray

    depth_m: float
    depth_valid: bool
    current_depth_valid: bool
    depth_valid_ratio: float
    depth_std_m: float

    age_frames: int
    hit_count: int
    consecutive_hits: int
    missed_frames: int

    track_state: int
    is_confirmed: bool
    is_currently_detected: bool
    actionable: bool

    mask: Optional[np.ndarray] = None


class _TrackInternal:
    """Internal helper maintaining state buffers and lifecycle metrics for a single track ID."""

    def __init__(self, track_id: int, initial_det, depth_est, class_history_size=5, depth_history_size=5):
        self.track_id = track_id
        self.age_frames = 1
        self.hit_count = 1
        self.consecutive_hits = 1
        self.missed_frames = 0
        self.is_confirmed = False

        self.bbox_xyxy = np.asarray(initial_det.box_xyxy, dtype=np.float32).copy()
        
        # Use candidate root if supplied or initial root_point
        cand_root = getattr(initial_det, "_cand_root", initial_det.root_point)
        self.root_point = np.asarray(cand_root, dtype=np.float32).copy()

        self.class_history: deque = deque(maxlen=class_history_size)
        self.class_history.append((initial_det.class_id, initial_det.class_name, float(initial_det.confidence)))

        # Family stabilization state
        initial_family = 0 if initial_det.class_id in CROP_CLASSES else 2
        self.stable_family = initial_family
        self.pending_family = None
        self.pending_family_hits = 0

        # Depth history
        self.depth_history: deque = deque(maxlen=depth_history_size)
        self.current_depth_valid = bool(depth_est and depth_est.valid)
        if self.current_depth_valid:
            self.depth_history.append(float(depth_est.depth_m))

        self.last_depth_valid_ratio = float(depth_est.valid_ratio) if depth_est else 0.0
        self.last_depth_std_m = float(depth_est.std_m) if depth_est else 0.0
        self.mask = initial_det.mask if hasattr(initial_det, "mask") else None

    def update_observation(self, det, depth_est, bbox_alpha, root_alpha):
        self.age_frames += 1
        self.hit_count += 1
        self.consecutive_hits += 1
        self.missed_frames = 0

        # EMA bbox
        raw_bbox = np.asarray(det.box_xyxy, dtype=np.float32)
        self.bbox_xyxy = bbox_alpha * raw_bbox + (1.0 - bbox_alpha) * self.bbox_xyxy

        # Root point smoothing (Assign prefiltered root directly to avoid double EMA)
        if hasattr(det, "_cand_root") and det._cand_root is not None:
            self.root_point = np.asarray(det._cand_root, dtype=np.float32).copy()
        else:
            raw_root = np.asarray(det.root_point, dtype=np.float32)
            self.root_point = root_alpha * raw_root + (1.0 - root_alpha) * self.root_point

        # Class history & Family hysteresis update
        det_family = 0 if det.class_id in CROP_CLASSES else 2
        if det_family == self.stable_family:
            self.pending_family = None
            self.pending_family_hits = 0
        else:
            if self.pending_family == det_family:
                self.pending_family_hits += 1
            else:
                self.pending_family = det_family
                self.pending_family_hits = 1

            if self.pending_family_hits >= 3:
                self.stable_family = det_family
                self.pending_family = None
                self.pending_family_hits = 0

        self.class_history.append((det.class_id, det.class_name, float(det.confidence)))

        # Current frame depth validity
        self.current_depth_valid = bool(depth_est and depth_est.valid)
        if self.current_depth_valid:
            self.depth_history.append(float(depth_est.depth_m))

        if depth_est:
            self.last_depth_valid_ratio = float(depth_est.valid_ratio)
            self.last_depth_std_m = float(depth_est.std_m)

        self.mask = det.mask if hasattr(det, "mask") else None

    def update_missed(self):
        self.age_frames += 1
        self.missed_frames += 1
        self.consecutive_hits = 0
        self.current_depth_valid = False
        self.mask = None


class TrackStateManager:
    """Manages persistent track lifecycles, EMA smoothing, and state transitions."""

    def __init__(self,
                 minimum_confirm_hits: int = 3,
                 maximum_publish_misses: int = 2,
                 maximum_track_misses: int = 15,
                 bbox_ema_alpha: float = 0.50,
                 root_ema_alpha: float = 0.35,
                 class_history_size: int = 5,
                 depth_history_size: int = 5,
                 require_depth: bool = True):

        self.minimum_confirm_hits = minimum_confirm_hits
        self.maximum_publish_misses = maximum_publish_misses
        self.maximum_track_misses = maximum_track_misses
        # False on depth-less rigs (e.g. UVC global-shutter camera with no depth
        # stream) — downstream already falls back to config.FAKE_DEPTH_M for Z
        # in that case, so actionability shouldn't depend on current_depth_valid.
        self.require_depth = require_depth
        self.bbox_ema_alpha = bbox_ema_alpha
        self.root_ema_alpha = root_ema_alpha
        self.class_history_size = class_history_size
        self.depth_history_size = depth_history_size

        self.tracks: Dict[int, _TrackInternal] = {}

    def predict_filtered_root(self, track_id: int, raw_root: np.ndarray) -> np.ndarray:
        """Predict candidate filtered root location prior to depth sampling."""
        raw_pt = np.asarray(raw_root, dtype=np.float32)
        if track_id in self.tracks:
            prev_root = self.tracks[track_id].root_point
            return self.root_ema_alpha * raw_pt + (1.0 - self.root_ema_alpha) * prev_root
        return raw_pt

    def update(self, detections: List, depth_estimates: Optional[Dict[int, any]] = None) -> List[PlantTrackState]:
        """
        Update states using current detections and depth estimates.
        Returns list of active PlantTrackStates.
        """
        if depth_estimates is None:
            depth_estimates = {}

        seen_track_ids = set()

        for det in detections:
            t_id = getattr(det, "track_id", None)
            if t_id is None or t_id < 0:
                continue

            seen_track_ids.add(t_id)
            depth_est = depth_estimates.get(t_id, None)

            if t_id in self.tracks:
                self.tracks[t_id].update_observation(
                    det, depth_est, self.bbox_ema_alpha, self.root_ema_alpha
                )
            else:
                self.tracks[t_id] = _TrackInternal(
                    t_id, det, depth_est, self.class_history_size, self.depth_history_size
                )

        # Update missed tracks
        dead_ids = []
        for t_id, trk in self.tracks.items():
            if t_id not in seen_track_ids:
                trk.update_missed()
                if trk.missed_frames > self.maximum_track_misses:
                    dead_ids.append(t_id)

        for t_id in dead_ids:
            del self.tracks[t_id]

        return self.get_active_states()

    def get_active_states(self) -> List[PlantTrackState]:
        """Construct list of active PlantTrackState instances to publish."""
        states = []
        for t_id, trk in self.tracks.items():
            # Confirmation requires consecutive hits
            if trk.consecutive_hits >= self.minimum_confirm_hits:
                trk.is_confirmed = True

            is_detected = (trk.missed_frames == 0)

            # Determine track_state code
            if not trk.is_confirmed:
                state_code = TENTATIVE
            elif is_detected:
                state_code = CONFIRMED
            else:
                state_code = COASTING

            # Filter publishable tracks: detected OR coasting within maximum_publish_misses
            if not is_detected and (not trk.is_confirmed or trk.missed_frames > self.maximum_publish_misses):
                continue

            # Class family & subtype stabilization
            stable_class_id, stable_class_name, avg_conf = self._stabilize_class(trk)

            # Depth calculation:
            # depth_valid requires current frame to have valid depth AND current detection
            if is_detected and trk.current_depth_valid:
                depth_valid = True
                depth_m = float(np.median(trk.depth_history)) if trk.depth_history else -1.0
            else:
                depth_valid = False
                depth_m = float(np.median(trk.depth_history)) if trk.depth_history else -1.0

            # Strict actionability rule: current frame must have valid depth,
            # unless this rig has no depth stream at all (require_depth=False),
            # in which case downstream falls back to a fixed fake depth.
            actionable = (
                trk.is_confirmed
                and is_detected
                and (trk.current_depth_valid or not self.require_depth)
                and trk.missed_frames == 0
            )

            state = PlantTrackState(
                track_id=int(trk.track_id),
                class_id=int(stable_class_id),
                class_name=str(stable_class_name),
                confidence=float(avg_conf),
                bbox_xyxy=np.copy(trk.bbox_xyxy),
                root_point=np.copy(trk.root_point),
                depth_m=depth_m,
                depth_valid=bool(depth_valid),
                current_depth_valid=bool(is_detected and trk.current_depth_valid),
                depth_valid_ratio=float(trk.last_depth_valid_ratio) if is_detected else 0.0,
                depth_std_m=float(trk.last_depth_std_m) if is_detected else 0.0,
                age_frames=int(trk.age_frames),
                hit_count=int(trk.hit_count),
                consecutive_hits=int(trk.consecutive_hits),
                missed_frames=int(trk.missed_frames),
                track_state=int(state_code),
                is_confirmed=bool(trk.is_confirmed),
                is_currently_detected=bool(is_detected),
                actionable=bool(actionable),
                mask=trk.mask if is_detected else None,
            )
            states.append(state)

        return states

    @staticmethod
    def _stabilize_class(trk: _TrackInternal) -> Tuple[int, str, float]:
        """Two-level class stabilization: Family hysteresis -> Subtype voting."""
        if not trk.class_history:
            return 0, "crop_small_leaf", 0.0

        target_family_set = CROP_CLASSES if trk.stable_family == 0 else WEED_CLASSES

        # Filter class history by stable family
        family_history = [item for item in trk.class_history if item[0] in target_family_set]
        if not family_history:
            family_history = list(trk.class_history)

        scores: Dict[Tuple[int, str], float] = {}
        counts: Dict[Tuple[int, str], int] = {}

        for cls_id, cls_name, conf in family_history:
            key = (cls_id, cls_name)
            scores[key] = scores.get(key, 0.0) + conf
            counts[key] = counts.get(key, 0) + 1

        best_key = max(scores.keys(), key=lambda k: scores[k])
        avg_conf = scores[best_key] / counts[best_key]
        return best_key[0], best_key[1], avg_conf
