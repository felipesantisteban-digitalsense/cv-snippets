from typing import Dict, Tuple, List

BBox = Tuple[int, int, int, int]

class ExtrapolateBBoxForTracker:
    def __init__(self, max_missing_frames: int = 15):
        self.max_missing_frames = max_missing_frames
        # Guardamos bbox, el frame y la velocidad (dx, dy, dw, dh)
        self._tracks: Dict[int, Dict] = {}

    def update(self, frame_idx: int, detections: List[Tuple[int, BBox]]) -> None:
        seen_ids = set()
        for track_id, bbox in detections:
            seen_ids.add(track_id)
            velocity = (0, 0, 0, 0)
            
            if track_id in self._tracks:
                prev = self._tracks[track_id]
                dt = frame_idx - prev["last_seen"]
                if dt > 0:
                    # Calculamos velocidad: (actual - anterior) / tiempo
                    velocity = tuple((curr - last) / dt for last, curr in zip(prev["bbox"], bbox))

            self._tracks[track_id] = {
                "bbox": bbox,
                "velocity": velocity,
                "last_seen": frame_idx
            }

    def active_tracks(self, frame_idx: int) -> List[Tuple[int, BBox, bool]]:
        active = []
        expired_ids = []

        for track_id, data in self._tracks.items():
            age = frame_idx - data["last_seen"]

            if age == 0:
                active.append((track_id, data["bbox"], False))
            elif age <= self.max_missing_frames:
                # EXTRAPOLACIÓN: BBox_actual = BBox_last + (Velocidad * tiempo_transcurrido)
                pred_bbox = tuple(
                    int(pos + vel * age) 
                    for pos, vel in zip(data["bbox"], data["velocity"])
                )
                active.append((track_id, pred_bbox, True))
            else:
                expired_ids.append(track_id)

        for tid in expired_ids:
            del self._tracks[tid]
            
        return active