"""Enhanced object tracking for detected road signs with advanced features."""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import time
import numpy as np

if TYPE_CHECKING:
    from .types import Detection


@dataclass
class TrackedObject:
    """Represents a tracked object across frames with enhanced tracking metadata."""
    track_id: int
    label: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 0
    confidence_history: List[float] = field(default_factory=list)
    bbox_history: List[tuple] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy)
    avg_confidence: float = 0.0
    
    def update_history(self, confidence: float, bbox: tuple, max_history: int = 10):
        """Update tracking history and calculate statistics."""
        self.confidence_history.append(confidence)
        self.bbox_history.append(bbox)
        
        # Keep only recent history
        if len(self.confidence_history) > max_history:
            self.confidence_history = self.confidence_history[-max_history:]
        if len(self.bbox_history) > max_history:
            self.bbox_history = self.bbox_history[-max_history:]
        
        # Calculate average confidence
        self.avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        
        # Calculate velocity (center displacement)
        if len(self.bbox_history) >= 2:
            prev_center = self._get_center(self.bbox_history[-2])
            curr_center = self._get_center(self.bbox_history[-1])
            self.velocity = (curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
    
    @staticmethod
    def _get_center(bbox: tuple) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class EnhancedTracker:
    """
    Enhanced tracker for road signs with IoU-based matching, Kalman filtering hints,
    and velocity prediction for improved tracking accuracy.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_stable: int = 3, 
                 use_prediction: bool = True, confidence_decay: float = 0.95):
        """
        Initialize enhanced tracker.
        
        Args:
            iou_threshold: Minimum IoU to consider a match
            max_age: Maximum frames to keep a track without updates
            min_stable: Minimum number of frames for a track to be considered stable
            use_prediction: Enable velocity-based position prediction
            confidence_decay: Decay factor for tracks without updates
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_stable = min_stable
        self.use_prediction = use_prediction
        self.confidence_decay = confidence_decay
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections: List) -> List:
        """
        Update tracks with new detections using enhanced matching.
        
        Args:
            detections: List of Detection objects
        
        Returns:
            List of Detection objects with updated track_id fields
        """
        current_time = time.time()
        self.frame_count += 1
        
        # Predict positions for existing tracks
        if self.use_prediction:
            self._predict_positions()
        
        # Remove old tracks
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if (current_time - track.last_seen) < self.max_age
        }
        
        if not detections:
            return []
        
        # Build cost matrix for Hungarian algorithm (simplified greedy matching)
        cost_matrix = self._build_cost_matrix(detections)
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        updated_detections = []
        
        # Greedy matching: sort by IoU score (best matches first)
        matches = []
        for det_idx, detection in enumerate(detections):
            bbox = detection.bbox
            label = detection.label
            confidence = detection.confidence
            
            best_match = None
            best_score = self.iou_threshold
            
            # Find best matching track
            for tid, track in self.tracks.items():
                if tid in matched_tracks:
                    continue
                
                # Skip if labels don't match
                if track.label != label:
                    continue
                
                # Calculate IoU score
                iou = self._calculate_iou(bbox, track.bbox)
                
                # Apply confidence weighting
                score = iou * 0.7 + (confidence * track.avg_confidence) * 0.3
                
                if score > best_score:
                    best_score = score
                    best_match = tid
            
            if best_match is not None:
                matches.append((det_idx, best_match, best_score))
        
        # Sort matches by score (descending)
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Apply matches
        for det_idx, track_id, score in matches:
            if det_idx in matched_detections or track_id in matched_tracks:
                continue
            
            detection = detections[det_idx]
            track = self.tracks[track_id]
            
            # Update existing track
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.last_seen = current_time
            track.frame_count += 1
            track.update_history(detection.confidence, detection.bbox)
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)
            
            # Update detection with track_id
            detection.track_id = track_id
            updated_detections.append(detection)
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx in matched_detections:
                continue
            
            new_track = TrackedObject(
                track_id=self.next_id,
                label=detection.label,
                confidence=detection.confidence,
                bbox=detection.bbox,
                last_seen=current_time,
                frame_count=1
            )
            new_track.update_history(detection.confidence, detection.bbox)
            self.tracks[self.next_id] = new_track
            
            # Update detection with new track_id
            detection.track_id = self.next_id
            updated_detections.append(detection)
            self.next_id += 1
        
        # Apply confidence decay to unmatched tracks
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.confidence *= self.confidence_decay
        
        return updated_detections
    
    def _predict_positions(self):
        """Predict future positions of tracks based on velocity."""
        for track in self.tracks.values():
            if track.frame_count > 1 and track.velocity != (0.0, 0.0):
                x1, y1, x2, y2 = track.bbox
                vx, vy = track.velocity
                # Simple linear prediction
                track.bbox = (x1 + vx, y1 + vy, x2 + vx, y2 + vy)
    
    def _build_cost_matrix(self, detections: List) -> np.ndarray:
        """
        Build cost matrix for detection-track matching.
        
        Args:
            detections: List of Detection objects
        
        Returns:
            Cost matrix (lower is better)
        """
        n_detections = len(detections)
        n_tracks = len(self.tracks)
        
        if n_detections == 0 or n_tracks == 0:
            return np.array([])
        
        cost_matrix = np.zeros((n_detections, n_tracks))
        
        for i, detection in enumerate(detections):
            for j, (tid, track) in enumerate(self.tracks.items()):
                if detection.label != track.label:
                    cost_matrix[i, j] = 1.0  # High cost for label mismatch
                else:
                    iou = self._calculate_iou(detection.bbox, track.bbox)
                    cost_matrix[i, j] = 1.0 - iou  # Convert IoU to cost
        
        return cost_matrix
    
    @staticmethod
    def _calculate_iou(box1: tuple, box2: tuple) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
        
        Returns:
            IoU score between 0.0 and 1.0
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def _calculate_center_distance(box1: tuple, box2: tuple) -> float:
        """Calculate Euclidean distance between box centers."""
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2
        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2
        
        return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
    
    def is_stable(self, track_id: int) -> bool:
        """
        Check if a track is stable (has been tracked for min_stable frames).
        
        Args:
            track_id: The track ID to check
        
        Returns:
            True if the track is stable, False otherwise
        """
        if track_id not in self.tracks:
            return False
        track = self.tracks[track_id]
        return track.frame_count >= self.min_stable and track.avg_confidence >= 0.5
    
    def get_stable_tracks(self) -> List[TrackedObject]:
        """
        Get all stable tracks with high confidence.
        
        Returns:
            List of TrackedObject instances that are stable
        """
        return [
            track for track in self.tracks.values() 
            if track.frame_count >= self.min_stable and track.avg_confidence >= 0.5
        ]
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """
        Get a specific track by ID.
        
        Args:
            track_id: Track ID to retrieve
        
        Returns:
            TrackedObject if found, None otherwise
        """
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> List[TrackedObject]:
        """Get all active tracks."""
        return list(self.tracks.values())
    
    def get_track_count(self) -> int:
        """Get the number of active tracks."""
        return len(self.tracks)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking metrics
        """
        total_tracks = len(self.tracks)
        stable_tracks = len(self.get_stable_tracks())
        
        avg_confidence = 0.0
        avg_frame_count = 0.0
        if total_tracks > 0:
            avg_confidence = sum(t.avg_confidence for t in self.tracks.values()) / total_tracks
            avg_frame_count = sum(t.frame_count for t in self.tracks.values()) / total_tracks
        
        return {
            'total_tracks': total_tracks,
            'stable_tracks': stable_tracks,
            'avg_confidence': avg_confidence,
            'avg_frame_count': avg_frame_count,
            'frame_count': self.frame_count,
            'next_id': self.next_id
        }
    
    def reset(self):
        """Reset the tracker, clearing all tracks and statistics."""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0


# Backward compatibility alias
SimpleTracker = EnhancedTracker