from typing import Dict, Tuple, List
from inference.video.last_bbox_for_tracker import LastBBoxForTracker


BBox = Tuple[int, int, int, int]


class InterpolatedBBoxForTracker(LastBBoxForTracker):
    """
    Extends LastBBoxForTracker by linearly interpolating bounding boxes
    when detections reappear after short gaps.

    Args:
        max_missing_frames (int): Maximum number of frames to interpolate over.

    """

    def __init__(self, max_missing_frames: int = 15):
        super().__init__(max_missing_frames)
        self._interp: Dict[int, Dict] = {}

    def update(
        self,
        frame_idx: int,
        detections: List[Tuple[int, BBox]],
    ) -> None:
        seen_ids = set()

        for track_id, bbox in detections:
            seen_ids.add(track_id)

            if track_id in self._tracks:
                prev_data = self._tracks[track_id]
                gap = frame_idx - prev_data["last_seen"]

                if 1 < gap <= self.max_missing_frames:
                    self._interp[track_id] = {
                        "start_frame": prev_data["last_seen"],
                        "end_frame": frame_idx,
                        "from_bbox": prev_data["bbox"],
                        "to_bbox": bbox,
                    }

            self._tracks[track_id] = {
                "bbox": bbox,
                "last_seen": frame_idx,
            }

    def active_tracks(
        self,
        frame_idx: int,
    ) -> List[Tuple[int, BBox, bool]]:
        active = []
        expired_ids = []

        for track_id, data in self._tracks.items():
            age = frame_idx - data["last_seen"]

            # Real detection
            if age == 0:
                active.append((track_id, data["bbox"], False))
                continue

            # Interpolation
            if track_id in self._interp:
                interp = self._interp[track_id]

                if frame_idx <= interp["end_frame"]:
                    t0 = interp["start_frame"]
                    t1 = interp["end_frame"]
                    alpha = (frame_idx - t0) / max(1, (t1 - t0))

                    bbox = lerp_bbox(
                        interp["from_bbox"],
                        interp["to_bbox"],
                        alpha,
                    )
                    active.append((track_id, bbox, True))
                    continue
                else:
                    del self._interp[track_id]

            if age <= self.max_missing_frames:
                active.append((track_id, data["bbox"], True))
            else:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            self._tracks.pop(track_id, None)
            self._interp.pop(track_id, None)

        return active
    
def lerp_bbox(a, b, alpha):
    return tuple(
        int(x + alpha * (y - x))
        for x, y in zip(a, b)
    )