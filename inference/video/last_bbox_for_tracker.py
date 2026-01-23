from typing import Dict, Tuple, List

BBox = Tuple[int, int, int, int]

class LastBBoxForTracker:
    """
    Keeps drawing the last known bounding box for each track_id
    when detections are temporarily missing.

    Args:
        max_missing_frames: Maximum number of frames to keep drawing the last known bbox
            after the track was last seen.
    """

    def __init__(self, max_missing_frames: int = 15):
        self.max_missing_frames = max_missing_frames
        self._tracks: Dict[int, Dict] = {}

    def update(
        self,
        frame_idx: int,
        detections: List[Tuple[int, BBox]],
    ) -> None:
        """
        Update tracker state with detections from the current frame.

        Args:
            frame_idx: Current frame index.
            detections: List of (track_id, bbox) tuples.
        """
        seen_ids = set()

        for track_id, bbox in detections:
            self._tracks[track_id] = {
                "bbox": bbox,
                "last_seen": frame_idx,
            }
            seen_ids.add(track_id)


    def active_tracks(
        self,
        frame_idx: int,
    ) -> List[Tuple[int, BBox, bool]]:
        """
        Returns active tracks to be drawn.

        Returns:
            List of (track_id, bbox, is_fallback)
        """
        active = []

        expired_ids = []

        for track_id, data in self._tracks.items():
            age = frame_idx - data["last_seen"]

            if age == 0:
                active.append((track_id, data["bbox"], False))
            elif age <= self.max_missing_frames:
                active.append((track_id, data["bbox"], True))
            else:
                expired_ids.append(track_id)

        # Handle expired tracks
        for track_id in expired_ids:
            del self._tracks[track_id]

        return active