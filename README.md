# deep_sort_realtime-imx8mp
Modified deep_sort_realtime repo to suport on i.MX8 M Plus


```
Custom MobileNet tflite inference can be tested by setting four variables:
       1. embedder="mobilenet_imx_tflite_npu" and set the 
       2. Set the embedder_wts to the path of the mobilenet tflite
       3. set the embedder_model_name to path of vx_delegate_path (embedder_model_name is reused internally as vxdelegate path)
       4. Set embedder_gpu=False to ensure NPU execution.
 
tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3,
        nn_budget=None,
        embedder="mobilenet_imx_tflite_npu",   # mention the embedder as mobilenet_imx_tflite_npu
        embedder_wts="/home/scmd/person_tracking_poc/mobilenet_v2-224-140.tflite",  # mention the embedder_wts as path mobilenet tflite file
        embedder_model_name=VX_DELEGATE_PATH,  # path to the vx delegate path 
        embedder_gpu=False,                    # enable NPU execution
        bgr=True
    )
```

```
$cd ~/deep_sort_realtime-imx8mp
[scmd@nixos:~/deep_sort_realtime-imx8mp]$ python deepsort_test.py 
🎯 DeepSORT Standalone Testing Suite
============================================================
🧪 Testing DeepSORT Pipeline with Dummy Data
==================================================
1️⃣ Initializing DeepSORT tracker...
Loading TFLite MobileNet: mobilenet_v2-224-140.tflite
VX Delegate Path: /nix/store/96bsy96b042wsqgzazpdhcdkqhai9k7n-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so
INFO: Vx delegate: allowed_cache_mode set to 0.
INFO: Vx delegate: device num set to 0.
INFO: Vx delegate: allowed_builtin_code set to 0.
INFO: Vx delegate: error_during_init set to 0.
INFO: Vx delegate: error_during_prepare set to 0.
INFO: Vx delegate: error_during_invoke set to 0.
✅ TFLite MobileNet NPU acceleration enabled
TFLite MobileNet input size: (224, 224)
W [HandleLayoutInfer:317]Op 162: default layout inference pass.
W [HandleLayoutInfer:317]Op 162: default layout inference pass.
W [HandleLayoutInfer:317]Op 162: default layout inference pass.
✅ Tracker initialized successfully

2️⃣ Creating dummy frame...
✅ Created dummy frame: (480, 640, 3)

3️⃣ Creating dummy detections...
✅ Created 3 dummy detections
   Detection 1: [100, 50, 80, 150] conf=0.90000000000000002 class=person
   Detection 2: [300, 100, 75, 140] conf=0.80000000000000004 class=person
   Detection 3: [500, 80, 70, 130] conf=0.84999999999999998 class=person

4️⃣ Testing tracking over multiple frames...

--- Frame 1 ---
Processing 3 detections...
✅ Frame 1 processed successfully!
   📊 Total tracks: 3
   ✅ Confirmed tracks: 0
   ⏳ Tentative tracks: 3

--- Frame 2 ---
Processing 3 detections...
✅ Frame 2 processed successfully!
   📊 Total tracks: 3
   ✅ Confirmed tracks: 0
   ⏳ Tentative tracks: 3

--- Frame 3 ---
Processing 3 detections...
✅ Frame 3 processed successfully!
   📊 Total tracks: 3
   ✅ Confirmed tracks: 3
   ⏳ Tentative tracks: 0
   🎯 Track ID 1: bbox=[102, 51, 182, 201] age=3 hits=3
   🎯 Track ID 2: bbox=[300, 97, 375, 237] age=3 hits=3
   🎯 Track ID 3: bbox=[501, 79, 571, 209] age=3 hits=3

--- Frame 4 ---
Processing 3 detections...
✅ Frame 4 processed successfully!
   📊 Total tracks: 3
   ✅ Confirmed tracks: 3
   ⏳ Tentative tracks: 0
   🎯 Track ID 1: bbox=[97, 50, 177, 200] age=4 hits=4
   🎯 Track ID 2: bbox=[298, 101, 373, 241] age=4 hits=4
   🎯 Track ID 3: bbox=[501, 81, 571, 211] age=4 hits=4

--- Frame 5 ---
Processing 3 detections...
✅ Frame 5 processed successfully!
   📊 Total tracks: 3
   ✅ Confirmed tracks: 3
   ⏳ Tentative tracks: 0
   🎯 Track ID 1: bbox=[96, 47, 176, 197] age=5 hits=5
   🎯 Track ID 2: bbox=[300, 100, 375, 240] age=5 hits=5
   🎯 Track ID 3: bbox=[499, 82, 569, 212] age=5 hits=5

==================================================
🎉 All tests passed! DeepSORT is working correctly!

📊 Test Summary:
✅ Tracker initialization: SUCCESS
✅ Dummy data creation: SUCCESS
✅ Feature extraction (NPU): SUCCESS
✅ Multi-frame tracking: SUCCESS
✅ Track management: SUCCESS
✅ Test 1: DeepSORT Pipeline - PASSED

🧪 Testing TFLite NPU Embedder Directly
==================================================
1️⃣ Creating embedder...
Loading TFLite MobileNet: mobilenet_v2-224-140.tflite
VX Delegate Path: /nix/store/96bsy96b042wsqgzazpdhcdkqhai9k7n-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so
INFO: Vx delegate: allowed_cache_mode set to 0.
INFO: Vx delegate: device num set to 0.
INFO: Vx delegate: allowed_builtin_code set to 0.
INFO: Vx delegate: error_during_init set to 0.
INFO: Vx delegate: error_during_prepare set to 0.
INFO: Vx delegate: error_during_invoke set to 0.
✅ TFLite MobileNet NPU acceleration enabled
TFLite MobileNet input size: (224, 224)
W [HandleLayoutInfer:317]Op 162: default layout inference pass.
W [HandleLayoutInfer:317]Op 162: default layout inference pass.
W [HandleLayoutInfer:317]Op 162: default layout inference pass.
✅ Embedder created successfully

2️⃣ Creating dummy image crops...
✅ Created 3 dummy crops

3️⃣ Extracting features...
✅ Feature extraction successful!
   📊 Features shape: (3, 1001)
   📏 Feature dimension: 1001 per person
   🔢 Features range: [0.012, 0.060]

4️⃣ Testing feature similarity...
   📐 Cosine similarity between crop 1 & 2: 0.998
   📏 Feature norms: 1.000, 1.000
✅ Embedder test completed successfully!
✅ Test 2: TFLite NPU Embedder - PASSED

============================================================
📊 FINAL TEST RESULTS
============================================================
Tests Passed: 2/2
🎉 ALL TESTS PASSED! Your DeepSORT setup is fully functional!

🚀 Ready for real-world usage:
   ✅ NPU acceleration working
   ✅ Feature extraction working
   ✅ Multi-object tracking working
   ✅ Track management working

💡 Next step: Integrate with your YOLO detector!

[scmd@nixos:~/deep_sort_realtime-imx8mp]$
