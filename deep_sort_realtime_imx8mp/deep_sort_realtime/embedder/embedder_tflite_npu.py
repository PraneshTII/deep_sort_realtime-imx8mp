import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class MobileNetTFLiteNPU_Embedder:
    """Custom Mobilenet TFLite embedder with NPU acceleration for deep-sort-realtime"""
    
    def __init__(self, model_path, vx_delegate_path=None, use_npu=True, bgr=True, max_batch_size=16):
        self.model_path = model_path
        self.bgr = bgr
        self.max_batch_size = max_batch_size
        
        # Set VX delegate path with fallback
        if vx_delegate_path is None:
            print("ERROR: VX delegate path is not passed, set embedder_model_name parameter in DeepSort function to path of vx_delegate")
            return
        else:
            self.vx_delegate_path = vx_delegate_path
        
        print(f"Loading TFLite MobileNet: {os.path.basename(model_path)}")
        print(f"VX Delegate Path: {self.vx_delegate_path}")
        
        # Create TFLite interpreter with VX delegate for NPU
        if use_npu and os.path.exists(self.vx_delegate_path):
            try:
                self.interpreter = tflite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[
                        tflite.load_delegate(self.vx_delegate_path)
                    ]
                )
                self.using_npu = True
                print("✅ TFLite MobileNet NPU acceleration enabled")
            except Exception as e:
                print(f"⚠ TFLite MobileNet NPU failed, using CPU: {e}")
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.using_npu = False
        else:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.using_npu = False
        
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        input_shape = self.input_details[0]['shape']
        self.input_size = (input_shape[2], input_shape[1])  # (width, height)
        print(f"TFLite MobileNet input size: {self.input_size}")
        
        # Warmup
        self._warmup()

    def _warmup(self):
        """Warmup the model"""
        dummy_input = np.random.randint(0, 255, self.input_details[0]['shape'], dtype=np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
        self.interpreter.invoke()

    def predict(self, np_imgs):
        """
        Predict embeddings for batch of images.
        Compatible with deep-sort-realtime interface.
        
        Args:
            np_imgs: List or numpy array of images (crops)
        
        Returns:
            numpy array of feature embeddings
        """
        if len(np_imgs) == 0:
            return np.array([])
        
        # Handle single image
        if isinstance(np_imgs, np.ndarray) and len(np_imgs.shape) == 3:
            np_imgs = [np_imgs]
        
        embeddings = []
        
        # Process images in batches
        for i in range(0, len(np_imgs), self.max_batch_size):
            batch = np_imgs[i:i + self.max_batch_size]
            batch_embeddings = self._process_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

    def _process_batch(self, batch_imgs):
        """Process a batch of images"""
        batch_embeddings = []
        
        for img in batch_imgs:
            # Convert BGR to RGB if needed
            if self.bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            resized = cv2.resize(img, self.input_size)
            
            # Preprocess based on model requirements
            if self.input_details[0]['dtype'] == np.uint8:
                input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
            else:
                input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get feature vector (from last layer before classification)
            features = self.interpreter.get_tensor(self.output_details[-1]['index'])
            features_flat = features.flatten()
            
            # Normalize features
            norm = np.linalg.norm(features_flat)
            if norm > 0:
                features_flat = features_flat / norm
            
            batch_embeddings.append(features_flat)
        
        return batch_embeddings
