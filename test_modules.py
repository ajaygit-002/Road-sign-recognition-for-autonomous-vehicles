"""Test script to verify controls.py, preprocess.py, and tracker.py are error-free."""

import numpy as np
from src.utils.controls import ControlState, ControlListener, SystemMode
from src.utils.preprocess import preprocess_frame
from src.utils.tracker import SimpleTracker, TrackedObject
from src.config import PreprocessConfig, TrackingConfig

def test_controls():
    """Test ControlState functionality."""
    print("Testing ControlState...")
    
    cs = ControlState()
    assert cs.request_quit == False
    assert cs.paused == False
    assert cs.manual_override == False
    
    # Test toggle pause
    cs.toggle_pause()
    assert cs.paused == True
    assert cs.mode == SystemMode.PAUSED
    
    # Test set quit
    cs.set_quit()
    assert cs.request_quit == True
    
    # Test manual override
    cs.set_manual_override(True)
    assert cs.manual_override == True
    
    # Test emergency stop
    cs.toggle_emergency_stop()
    assert cs.emergency_stop == True
    
    print("✓ ControlState tests passed")
    
    # Test ControlListener
    print("Testing ControlListener...")
    cs2 = ControlState()
    listener = ControlListener(cs2)
    listener.handle_key('p')  # Toggle pause
    assert cs2.paused == True
    listener.handle_key('q')  # Quit
    assert cs2.request_quit == True
    
    print("✓ ControlListener tests passed")


def test_preprocess():
    """Test preprocess_frame functionality."""
    print("\nTesting preprocess_frame...")
    
    # Test without config
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed = preprocess_frame(frame)
    assert processed.shape == (640, 640, 3)
    print("✓ Preprocess without config passed")
    
    # Test with config
    cfg = PreprocessConfig()
    processed = preprocess_frame(frame, cfg)
    assert processed.shape == (640, 640, 3)
    print("✓ Preprocess with config passed")
    
    # Test different config options
    cfg.enable_blur = True
    cfg.enable_sharpen = True
    cfg.enable_gamma = True
    processed = preprocess_frame(frame, cfg)
    assert processed.shape == (640, 640, 3)
    print("✓ Preprocess with advanced options passed")


def test_tracker():
    """Test EnhancedTracker functionality."""
    print("\nTesting EnhancedTracker...")
    
    from src.utils.types import Detection
    
    tracker = SimpleTracker(iou_threshold=0.3, max_age=30, min_stable=3)
    assert tracker.iou_threshold == 0.3
    assert tracker.max_age == 30
    assert tracker.min_stable == 3
    print("✓ Tracker initialization passed")
    
    # Test tracking with Detection objects
    detections = [
        Detection(label='stop_sign', confidence=0.9, bbox=(10, 10, 50, 50)),
        Detection(label='speed_limit', confidence=0.85, bbox=(100, 100, 150, 150))
    ]
    
    tracked = tracker.update(detections)
    assert len(tracked) == 2
    assert tracked[0].label == 'stop_sign'
    assert tracked[1].label == 'speed_limit'
    assert tracked[0].track_id == 0
    assert tracked[1].track_id == 1
    print("✓ Tracker update with Detection objects passed")
    
    # Test stable tracks
    for i in range(5):
        tracked = tracker.update(detections)
    
    stable_tracks = tracker.get_stable_tracks()
    assert len(stable_tracks) == 2
    assert tracker.is_stable(0) == True
    assert tracker.is_stable(1) == True
    print("✓ Stable tracks detection passed")
    
    # Test new features
    stats = tracker.get_statistics()
    assert stats['total_tracks'] == 2
    assert stats['stable_tracks'] == 2
    print("✓ Statistics feature passed")
    
    # Test reset
    tracker.reset()
    assert len(tracker.tracks) == 0
    assert tracker.next_id == 0
    print("✓ Tracker reset passed")


def test_integration():
    """Test integration of all modules."""
    print("\nTesting integration...")
    
    from src.config import AppConfig
    from src.utils.types import Detection
    
    cfg = AppConfig()
    ctrl = ControlState()
    
    # Create frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Preprocess
    processed = preprocess_frame(frame, cfg.preprocess)
    assert processed.shape == (640, 640, 3)
    
    # Initialize tracker
    tracker = SimpleTracker(
        cfg.tracking.iou_threshold,
        cfg.tracking.max_age,
        cfg.tracking.min_stable
    )
    
    # Simulate detections with Detection objects
    detections = [Detection(label='yield', confidence=0.95, bbox=(20, 20, 60, 60))]
    tracked = tracker.update(detections)
    
    # Control operations
    ctrl.toggle_pause()
    ctrl.set_quit()
    
    assert ctrl.paused == True
    assert ctrl.request_quit == True
    assert len(tracked) == 1
    assert tracked[0].track_id == 0
    
    print("✓ Integration test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running module tests...")
    print("=" * 60)
    
    try:
        test_controls()
        test_preprocess()
        test_tracker()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓✓✓ ALL TESTS PASSED - NO ERRORS FOUND ✓✓✓")
        print("=" * 60)
        print("\nModules tested:")
        print("  • src/utils/controls.py")
        print("  • src/utils/preprocess.py")
        print("  • src/utils/tracker.py")
        print("\nThe application is ready to run!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
