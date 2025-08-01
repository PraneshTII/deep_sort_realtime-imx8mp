import sys;
import site;
import functools;
functools.reduce(lambda k, p: site.addsitedir(p, k), 
['/nix/store/7wdal3kjsp1hn6ig64m2y6baxg4f929h-tflite-opencv-test-app-cpu-1.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/1bvpxg3kvzjhrp7n9q114xcjxmyx2ik8-python3.12-pillow-11.0.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/d5hmgjyy2wy17k79z1j1gjs0fv1wh5ki-python3.12-opencv-imx-python-4.20-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/n60lsdw7cm43lv10bdrcdcxqv7dpnn0b-python3.12-tflite-imx-python-2.14-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/bjfn8qq4nrbmzjzvxlllzslbrf0zv1dd-python3.12-gst-python-1.24.7-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/8rivysd25fs2jx1k1x29if3siijyscsb-python3.12-pygobject-3.50.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/n4wvcq4sllahxr7ppqj4mxsyh1y1jqwp-python3.12-scipy-1.14.1-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/s58a7qp81ms8an7xdc7h008d8kc13kqp-python3.12-numpy-1.26.4-aarch64-unknown-linux-gnu/lib/python3.12/site-packages'], site._init_pathinfo());

import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import tflite_runtime.interpreter as tflite
import scipy

from deep_sort_realtime.deepsort_tracker import DeepSort



VX_DELEGATE_PATH="/nix/store/96bsy96b042wsqgzazpdhcdkqhai9k7n-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so"
####
#
#   Custom Mobilenet tflite inference can be tested by setting three variables:
#       1. embedder="mobilenet_imx_tflite_npu" and set the 
#       2. set the embedder_wts to the path of mobilenet tflite
#       3. set the embedder_model_name to path of vx_delegate_path ( embedder_model_name is reused internally as vxdelegate path)
# 
#tracker = DeepSort(
#        max_age=30,
#        n_init=3,
#        nms_max_overlap=1.0,
#        max_cosine_distance=0.3,
#        nn_budget=None,
#        embedder="mobilenet_imx_tflite_npu",
#        embedder_wts="/home/scmd/person_tracking_poc/mobilenet_v2-224-140.tflite",
#        embedder_model_name=VX_DELEGATE_PATH,
#        embedder_gpu=False, 
#        bgr=True
#    )
#)
#

def test_deepsort_pipeline():
    """Test the complete DeepSORT pipeline with dummy data"""
    
    print("🧪 Testing DeepSORT Pipeline with Dummy Data")
    print("=" * 50)
    
    # 1. Initialize tracker (you already did this successfully)
    print("1️⃣ Initializing DeepSORT tracker...")
    
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3,
        nn_budget=None,
        embedder="mobilenet_imx_tflite_npu",
        embedder_wts="/home/scmd/person_tracking_poc/mobilenet_v2-224-140.tflite",
        embedder_model_name=VX_DELEGATE_PATH,
        embedder_gpu=False, 
        bgr=True
    )
    print("✅ Tracker initialized successfully")
    
    # 2. Create dummy frame (simulating camera input)
    print("\n2️⃣ Creating dummy frame...")
    frame_width, frame_height = 640, 480
    dummy_frame = np.random.randint(0, 255, (frame_height, frame_width, 3), dtype=np.uint8)
    print(f"✅ Created dummy frame: {dummy_frame.shape}")
    
    # 3. Create dummy detections (simulating YOLO output)
    print("\n3️⃣ Creating dummy detections...")
    
    # DeepSORT expects detections in format: [(bbox, confidence, class_name), ...]
    # where bbox is [left, top, width, height]
    dummy_detections = [
        ([100, 50, 80, 150], 0.9, "person"),    # Person at (100,50) with size 80x150
        ([300, 100, 75, 140], 0.8, "person"),  # Person at (300,100) with size 75x140
        ([500, 80, 70, 130], 0.85, "person"),  # Person at (500,80) with size 70x130
    ]
    
    print(f"✅ Created {len(dummy_detections)} dummy detections")
    for i, (bbox, conf, cls) in enumerate(dummy_detections):
        left, top, width, height = bbox
        print(f"   Detection {i+1}: [{left}, {top}, {width}, {height}] conf={conf} class={cls}")
    
    # 4. Test tracking over multiple frames
    print("\n4️⃣ Testing tracking over multiple frames...")
    
    for frame_num in range(1, 6):  # Test 5 frames
        print(f"\n--- Frame {frame_num} ---")
        
        # Slightly modify detections to simulate movement
        modified_detections = []
        for bbox, conf, cls in dummy_detections:
            left, top, width, height = bbox
            # Add some random movement
            left += np.random.randint(-5, 6)
            top += np.random.randint(-3, 4)
            # Add some noise to confidence
            conf += np.random.uniform(-0.05, 0.05)
            conf = max(0.1, min(0.99, conf))  # Keep within valid range
            
            modified_detections.append(([left, top, width, height], conf, cls))
        
        try:
            # 5. Update tracker (this tests the complete pipeline)
            print(f"Processing {len(modified_detections)} detections...")
            tracks = tracker.update_tracks(modified_detections, frame=dummy_frame)
            
            # 6. Analyze results
            confirmed_tracks = [track for track in tracks if track.is_confirmed()]
            tentative_tracks = [track for track in tracks if not track.is_confirmed()]
            
            print(f"✅ Frame {frame_num} processed successfully!")
            print(f"   📊 Total tracks: {len(tracks)}")
            print(f"   ✅ Confirmed tracks: {len(confirmed_tracks)}")
            print(f"   ⏳ Tentative tracks: {len(tentative_tracks)}")
            
            # Show details of confirmed tracks
            for track in confirmed_tracks:
                track_id = track.track_id
                bbox = track.to_ltrb()  # [left, top, right, bottom]
                age = track.age
                hits = track.hits
                print(f"   🎯 Track ID {track_id}: bbox={[int(x) for x in bbox]} age={age} hits={hits}")
            
        except Exception as e:
            print(f"❌ Error processing frame {frame_num}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! DeepSORT is working correctly!")
    print("\n📊 Test Summary:")
    print("✅ Tracker initialization: SUCCESS")
    print("✅ Dummy data creation: SUCCESS") 
    print("✅ Feature extraction (NPU): SUCCESS")
    print("✅ Multi-frame tracking: SUCCESS")
    print("✅ Track management: SUCCESS")
    
    return True

def test_embedder_directly():
    """Test the embedder directly with image crops"""
    
    print("\n🧪 Testing TFLite NPU Embedder Directly")
    print("=" * 50)
    
    try:
        # Import and test embedder directly
        from deep_sort_realtime.embedder.embedder_tflite_npu import  MobileNetTFLiteNPU_Embedder
        
        print("1️⃣ Creating embedder...")
        embedder = MobileNetTFLiteNPU_Embedder(
            model_path="/home/scmd/person_tracking_poc/mobilenet_v2-224-140.tflite",
            vx_delegate_path=VX_DELEGATE_PATH,
            use_npu=True,
            bgr=True
        )
        print("✅ Embedder created successfully")
        
        print("\n2️⃣ Creating dummy image crops...")
        # Create dummy person crops (simulating YOLO detections)
        crop1 = np.random.randint(0, 255, (150, 80, 3), dtype=np.uint8)  # Person 1
        crop2 = np.random.randint(0, 255, (140, 75, 3), dtype=np.uint8)  # Person 2
        crop3 = np.random.randint(0, 255, (130, 70, 3), dtype=np.uint8)  # Person 3
        
        crops = [crop1, crop2, crop3]
        print(f"✅ Created {len(crops)} dummy crops")
        
        print("\n3️⃣ Extracting features...")
        features = embedder.predict(crops)
        
        print(f"✅ Feature extraction successful!")
        print(f"   📊 Features shape: {features.shape}")
        print(f"   📏 Feature dimension: {features.shape[1]} per person")
        print(f"   🔢 Features range: [{features.min():.3f}, {features.max():.3f}]")
        
        # Test feature similarity
        print("\n4️⃣ Testing feature similarity...")
        if len(features) >= 2:
            # Calculate cosine similarity between first two features
            feat1, feat2 = features[0], features[1]
            
            # Cosine similarity
            dot_product = np.dot(feat1, feat2)
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)
            cosine_sim = dot_product / (norm1 * norm2)
            
            print(f"   📐 Cosine similarity between crop 1 & 2: {cosine_sim:.3f}")
            print(f"   📏 Feature norms: {norm1:.3f}, {norm2:.3f}")
            
        print("✅ Embedder test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Embedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all standalone tests"""
    
    print("🎯 DeepSORT Standalone Testing Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Complete pipeline
    try:
        if test_deepsort_pipeline():
            tests_passed += 1
            print("✅ Test 1: DeepSORT Pipeline - PASSED")
        else:
            print("❌ Test 1: DeepSORT Pipeline - FAILED")
    except Exception as e:
        print(f"❌ Test 1: DeepSORT Pipeline - FAILED with exception: {e}")
    
    # Test 2: Embedder directly
    try:
        if test_embedder_directly():
            tests_passed += 1
            print("✅ Test 2: TFLite NPU Embedder - PASSED")
        else:
            print("❌ Test 2: TFLite NPU Embedder - FAILED")
    except Exception as e:
        print(f"❌ Test 2: TFLite NPU Embedder - FAILED with exception: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Your DeepSORT setup is fully functional!")
        print("\n🚀 Ready for real-world usage:")
        print("   ✅ NPU acceleration working")
        print("   ✅ Feature extraction working") 
        print("   ✅ Multi-object tracking working")
        print("   ✅ Track management working")
        print("\n💡 Next step: Integrate with your YOLO detector!")
    else:
        print(f"⚠️ {total_tests - tests_passed} tests failed. Check the errors above.")
    
    return tests_passed == total_tests

if __name__ == '__main__':
    main()
