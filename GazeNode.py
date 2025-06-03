# ComfyUI/custom_nodes/ComfyUI-Gaze/GazeNode.py

import torch
import numpy as np
# from PIL import Image # If not used directly, can be removed
import os
import shutil # For potentially moving user-provided models to a standard location

import comfy.model_management
import comfy.utils
import folder_paths # ComfyUI's way to get model paths

# --- Define where Gaze Predictor models are stored within ComfyUI/models ---
GAZE_MODEL_SUBDIR_NAME = "gaze" # Store gaze models in ComfyUI/models/gaze/
GAZE_PREDICTOR_FILENAME = "res18_x128_all_vfhq_vert.pth" # Your actual model filename

# --- Global Cache for Gaze Models ---
GAZE_DETECTOR_INSTANCE = None # For InsightFace detector
GAZE_PREDICTOR_INSTANCE = None # For your Gaze Predictor model
CURRENT_GAZE_CONFIG = {} # Stores {"device": "cuda/cpu", "detector_threshold": float}

# --- Utility functions (average_output) ---
# (Keep average_output_for_node definition as previously discussed)
def average_output_for_node(out_dict, prev_dict):
    # ... (full implementation of average_output_for_node)
    # smooth gaze
    if prev_dict and 'gaze_out' in prev_dict and out_dict.get('gaze_out') is not None:
        gaze_out_current = np.array(out_dict['gaze_out'])
        gaze_out_prev = np.array(prev_dict['gaze_out'])
        if gaze_out_current.shape == gaze_out_prev.shape:
            out_dict['gaze_out'] = gaze_out_current + gaze_out_prev
            norm = np.linalg.norm(out_dict['gaze_out'])
            if norm > 1e-6:
                 out_dict['gaze_out'] /= norm
            else: 
                out_dict['gaze_out'] = gaze_out_current 
    # Full logic for smoothing verts_eyes
    if out_dict.get('verts_eyes') is not None and prev_dict and prev_dict.get('verts_eyes') is not None and \
       isinstance(out_dict['verts_eyes'], dict) and isinstance(prev_dict['verts_eyes'], dict) and \
       out_dict['verts_eyes'].get('left') is not None and prev_dict['verts_eyes'].get('left') is not None and \
       out_dict['verts_eyes'].get('right') is not None and prev_dict['verts_eyes'].get('right') is not None:
        if 'iris_idxs' in out_dict and 'centers_iris' in out_dict and \
           isinstance(out_dict['centers_iris'], dict) and \
           out_dict.get('iris_idxs') is not None and \
           out_dict['centers_iris'].get('left') is not None and \
           out_dict['centers_iris'].get('right') is not None:
            iris_idxs = np.array(out_dict['iris_idxs']) 
            if iris_idxs.size > 0: 
                max_iris_idx = np.max(iris_idxs)
                verts_left_current = np.array(out_dict['verts_eyes']['left'])
                verts_right_current = np.array(out_dict['verts_eyes']['right'])
                verts_left_prev = np.array(prev_dict['verts_eyes']['left'])
                verts_right_prev = np.array(prev_dict['verts_eyes']['right'])
                if max_iris_idx < len(verts_left_current) and max_iris_idx < len(verts_right_current):
                    norm_prev_left = np.linalg.norm(verts_left_prev)
                    norm_prev_right = np.linalg.norm(verts_right_prev)
                    if norm_prev_left > 1e-6 and norm_prev_right > 1e-6:
                        norm_curr_left = np.linalg.norm(verts_left_current)
                        norm_curr_right = np.linalg.norm(verts_right_current)
                        scale_l = norm_curr_left / norm_prev_left if norm_prev_left > 1e-6 else 1.0
                        scale_r = norm_curr_right / norm_prev_right if norm_prev_right > 1e-6 else 1.0
                        verts_left_current = verts_left_current * ((1 + (scale_l - 1) / 2) / (scale_l if scale_l > 1e-6 else 1.0))
                        verts_right_current = verts_right_current * ((1 + (scale_r - 1) / 2) / (scale_r if scale_r > 1e-6 else 1.0))
                        mean_left_iris = verts_left_current[iris_idxs][:, :2].mean(axis=0)
                        mean_right_iris = verts_right_current[iris_idxs][:, :2].mean(axis=0)
                        verts_left_current[:, :2] += -mean_left_iris + np.array(out_dict['centers_iris']['left'])
                        verts_right_current[:, :2] += -mean_right_iris + np.array(out_dict['centers_iris']['right'])
                        out_dict['verts_eyes']['left'] = verts_left_current
                        out_dict['verts_eyes']['right'] = verts_right_current
    return out_dict


# --- Dynamically Imported Components ---
GAZE_COMPONENTS_LOADED_SUCCESSFULLY = False
FaceDetector = None
GazePredictor = None
Timer = None
draw_results = None
draw_eyes = None
draw_gaze = None

try:
    from .models.face_detector import FaceDetectorIF as FaceDetector
    from .models.gaze_predictor import GazePredictorHandler as GazePredictor
    from .utils import Timer, draw_results, draw_eyes, draw_gaze
    GAZE_COMPONENTS_LOADED_SUCCESSFULLY = True
    print("✅ ComfyUI-GazeNode: Core gaze components imported successfully.")
except ImportError as e:
    # ... (error printing as before) ...
    print(f"ComfyUI-GazeNode: CRITICAL ERROR importing Gaze components: {e}.")
    print("Expected structure relative to GazeNode.py:")
    print("  ./models/__init__.py, ./models/face_detector.py, ./models/gaze_predictor.py")
    print("  ./utils/__init__.py, ./utils/timer.py, ./utils/utils.py")
    raise
# ... (other except block) ...

class GazeNode:
    def __init__(self):
        # ... (init as before) ...
        if not GAZE_COMPONENTS_LOADED_SUCCESSFULLY:
            print("ComfyUI-GazeNode: Node initialized, but core components failed to load earlier.")
        self.detector = None
        self.predictor = None


    @classmethod
    def INPUT_TYPES(cls): # Sames as before
        return {
            "required": {
                "images": ("IMAGE",),
                "face_detector_threshold": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.01}),
                "enable_eyes": ("BOOLEAN", {"default": True}), 
                "show_eyeball": ("BOOLEAN", {"default": True}), 
                "show_iris": ("BOOLEAN", {"default": True}), 
                "iris_mode": (["normal", "solid", "inverted"], {"default": "normal"}),  # Dropdown parameter   
                "squint_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),             
                "line_thickness": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 48.0, "step": 0.01}),                     
                "enable_gaze_vector": ("BOOLEAN", {"default": False}), 
                "enable_video": ("BOOLEAN", {"default": False}), 
                "smooth_predictions": ("BOOLEAN", {"default": True}),           
                "offload_models_on_finish": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_gaze_estimation"
    CATEGORY = "Gaze"


    def _get_gaze_predictor_model_path(self):

        base_models_dir = os.path.join(folder_paths.models_dir, GAZE_MODEL_SUBDIR_NAME)
        os.makedirs(base_models_dir, exist_ok=True) # Ensure the subdirectory exists

        # The full path to the specific model file
        # e.g., ComfyUI/models/gaze/res18_x128_all_vfhq_vert.pth
        specific_model_path = os.path.join(base_models_dir, GAZE_PREDICTOR_FILENAME)
        
        return specific_model_path

    def _ensure_gaze_predictor_model_exists(self):
        predictor_model_path = self._get_gaze_predictor_model_path()

        if not os.path.exists(predictor_model_path):
            # Attempt to find it in a bundled location within the custom node
            node_root_dir = os.path.dirname(__file__) # ComfyUI-Gaze/
            # Option 1: ComfyUI-Gaze/model_weights/your_model.pth
            bundled_path_mw = os.path.join(node_root_dir, "model_weights", GAZE_PREDICTOR_FILENAME)
            # Option 2: ComfyUI-Gaze/data/checkpoints/your_model.pth (as per your last path)
            bundled_path_dc = os.path.join(node_root_dir, "data", "checkpoints", GAZE_PREDICTOR_FILENAME)

            found_bundled_path = None
            if os.path.exists(bundled_path_mw):
                found_bundled_path = bundled_path_mw
            elif os.path.exists(bundled_path_dc):
                found_bundled_path = bundled_path_dc
            
            if found_bundled_path:
                print(f"ComfyUI-GazeNode: Found gaze predictor model at {found_bundled_path}.")
                print(f"ComfyUI-GazeNode: Copying to standard ComfyUI models location: {predictor_model_path}")
                try:
                    shutil.copy2(found_bundled_path, predictor_model_path)
                    print(f"ComfyUI-GazeNode: Model copied successfully.")
                except Exception as e:
                    raise RuntimeError(
                        f"ComfyUI-GazeNode: Failed to copy model from {found_bundled_path} to {predictor_model_path}. Error: {e}\n"
                        f"Please manually place '{GAZE_PREDICTOR_FILENAME}' in '{os.path.dirname(predictor_model_path)}'."
                    ) from e
            else:
                # If not found in bundled locations either, raise an error.
                raise FileNotFoundError(
                    f"ComfyUI-GazeNode: Gaze predictor model '{GAZE_PREDICTOR_FILENAME}' not found.\n"
                    f"Please place it in EITHER of these locations:\n"
                    f"1. {os.path.dirname(predictor_model_path)} (Recommended)\n"
                    f"2. {os.path.join(node_root_dir, 'model_weights')}\n"
                    f"3. {os.path.join(node_root_dir, 'data', 'checkpoints')}"
                )
        
        # print(f"ComfyUI-GazeNode: Gaze predictor model found at {predictor_model_path}")
        return predictor_model_path


    def _load_gaze_models(self, device_str_requested="cuda", detector_threshold=0.7):
        global GAZE_DETECTOR_INSTANCE, GAZE_PREDICTOR_INSTANCE, CURRENT_GAZE_CONFIG, FaceDetector, GazePredictor

        if FaceDetector is None or GazePredictor is None: # Should be caught by GAZE_COMPONENTS_LOADED_SUCCESSFULLY check
             raise RuntimeError("ComfyUI-GazeNode: Core FaceDetector or GazePredictor classes not imported.")

        target_device = comfy.model_management.get_torch_device() if device_str_requested == "cuda" else torch.device("cpu")

        # Check cache first
        if GAZE_DETECTOR_INSTANCE is not None and GAZE_PREDICTOR_INSTANCE is not None:
            if CURRENT_GAZE_CONFIG.get("device") == str(target_device) and \
               CURRENT_GAZE_CONFIG.get("detector_threshold") == detector_threshold:
                # Potentially move to device if necessary (though InsightFace might manage its own)
                # if hasattr(GAZE_DETECTOR_INSTANCE, 'to') and GAZE_DETECTOR_INSTANCE.device != target_device: GAZE_DETECTOR_INSTANCE.to(target_device)
                if hasattr(GAZE_PREDICTOR_INSTANCE, 'device') and GAZE_PREDICTOR_INSTANCE.device != target_device:
                    GAZE_PREDICTOR_INSTANCE.to(target_device) # Assuming predictor is a PyTorch model
                
                self.detector = GAZE_DETECTOR_INSTANCE
                self.predictor = GAZE_PREDICTOR_INSTANCE
                # print("ComfyUI-GazeNode: Reusing cached models.")
                return
            else:
                print(f"ComfyUI-GazeNode: Config changed. Reloading models.")
                # No need to nullify here, they will be replaced.

        # --- Load Face Detector (InsightFace) ---
        # This part doesn't need a specific model file path from us, InsightFace handles it.
        # We reload it if detector_threshold changes or if it's not loaded.
        if GAZE_DETECTOR_INSTANCE is None or CURRENT_GAZE_CONFIG.get("detector_threshold") != detector_threshold:
            print(f"ComfyUI-GazeNode: Initializing Face Detector (InsightFace)...")
            try:
                GAZE_DETECTOR_INSTANCE = FaceDetector(det_thresh=detector_threshold, det_size=640)
                # InsightFace handles device via 'providers' in its init.
                # if hasattr(GAZE_DETECTOR_INSTANCE, 'eval'): GAZE_DETECTOR_INSTANCE.eval()
                print(f"ComfyUI-GazeNode: Face Detector (InsightFace) initialized.")
            except Exception as e:
                print(f"ComfyUI-GazeNode: Failed to initialize Face Detector: {e}")
                GAZE_DETECTOR_INSTANCE = None # Ensure it's None if failed
                raise
        
        # --- Load Gaze Predictor ---
        # This needs the actual model file.
        if GAZE_PREDICTOR_INSTANCE is None or CURRENT_GAZE_CONFIG.get("device") != str(target_device): # Reload if device changed or not loaded
            print(f"ComfyUI-GazeNode: Loading Gaze Predictor model...")
            try:
                actual_predictor_model_path = self._ensure_gaze_predictor_model_exists()
                print(f"ComfyUI-GazeNode: Gaze Predictor model path: {actual_predictor_model_path}")
                
                # Use parameters from your YAML
                class MockPredictorConfig:
                    def __init__(self):
                        self.MODEL_PATH = actual_predictor_model_path # Corrected path
                        self.BACKBONE_TYPE = 'resnet' # From YAML
                        self.NUM_LAYERS = 18 # From YAML
                        self.IMAGE_SIZE = [128, 128] # From YAML
                        self.MODE = 'vertex' # From YAML
                        self.NUM_POINTS_OUT_EYES = 962 # From YAML
                        self.NUM_POINTS_OUT_FACE = 68 
                        self.BOUNDED = False # Default from config.py
                        self.EXPANSION = None # Default from config.py (though None might need handling if accessed)
                        self.EXTENT_TO_CROP_RATIO = 1.6 # Default from config.py
                        self.EXTENT_TO_CROP_RATIO_FACE = 2.0 # Default from config.py
                        self.PRETRAINED = actual_predictor_model_path

                predictor_config = MockPredictorConfig()
                GAZE_PREDICTOR_INSTANCE = GazePredictor(predictor_config, device=target_device)
                if hasattr(GAZE_PREDICTOR_INSTANCE, 'eval'): GAZE_PREDICTOR_INSTANCE.eval()
                print(f"ComfyUI-GazeNode: Gaze Predictor loaded to {target_device}.")
            except Exception as e:
                print(f"ComfyUI-GazeNode: Failed to load Gaze Predictor model: {e}")
                GAZE_PREDICTOR_INSTANCE = None
                raise

        self.detector = GAZE_DETECTOR_INSTANCE
        self.predictor = GAZE_PREDICTOR_INSTANCE
        
        CURRENT_GAZE_CONFIG = {
            "device": str(target_device),
            "detector_threshold": detector_threshold
        }
        # print(f"ComfyUI-GazeNode: Models configured for current run.")

    # tensor_to_cv2_list and cv2_list_to_tensor methods (same as before)
    def tensor_to_cv2_list(self, images: torch.Tensor) -> list:
        batch_size, height, width, channels = images.shape
        cv2_images = []
        for i in range(batch_size):
            np_image = (images[i].cpu().numpy() * 255).astype(np.uint8)
            cv2_images.append(np_image) 
        return cv2_images

    def cv2_list_to_tensor(self, cv2_images: list, device: torch.device) -> torch.Tensor:
        output_tensors = []
        for rgb_image_np in cv2_images: 
            tensor_image = torch.from_numpy(rgb_image_np.astype(np.float32) / 255.0)
            output_tensors.append(tensor_image)
        return torch.stack(output_tensors).to(device)

    def process_gaze_estimation(self, images: torch.Tensor, face_detector_threshold: float, 
                                enable_eyes: bool, show_eyeball: bool,show_iris: bool,iris_mode: str,squint_ratio: float,line_thickness:float, enable_gaze_vector: bool, enable_video: bool,
                                smooth_predictions: bool, offload_models_on_finish: bool):
        global Timer, draw_results, draw_eyes, draw_gaze # Ensure imported components are accessible




        if not GAZE_COMPONENTS_LOADED_SUCCESSFULLY: # Check the flag
            raise RuntimeError("ComfyUI-GazeNode: Core components not imported. Node cannot function.")
        if Timer is None or draw_eyes is None or draw_gaze is None: # Specific check
            raise RuntimeError("ComfyUI-GazeNode: Timer or drawing functions not imported correctly.")


        processing_device_type = "cuda" if comfy.model_management.get_torch_device().type == 'cuda' else "cpu"
        self._load_gaze_models(device_str_requested=processing_device_type, detector_threshold=face_detector_threshold)

        if self.detector is None or self.predictor is None:
            raise RuntimeError("ComfyUI-GazeNode: Gaze models could not be loaded. Check console for errors.")

        batch_input_rgb_np = self.tensor_to_cv2_list(images) 
        batch_output_rgb_np = []
        prev_gaze_dict = None 
        pbar = comfy.utils.ProgressBar(len(batch_input_rgb_np))
        # print(f"ComfyUI-GazeNode: Processing {len(batch_input_rgb_np)} images...") # Optional: for more verbose logging

        for i, img_rgb_np in enumerate(batch_input_rgb_np): 
            img_for_processing = img_rgb_np 
            processed_img_rgb_np = img_rgb_np.copy() 
            
            # Run face detection
            bboxes, lms5_all_faces, _ = self.detector.run(img_for_processing)
            
            current_gaze_dict = None # Initialize for current frame

            # --- Corrected logic for handling bboxes ---
            faces_found_and_valid = False
            if bboxes is not None and len(bboxes) > 0:
                # Check if the first element indicates a "no face" signal (integer 0)
                # or if it's a valid bounding box (NumPy array)
                if isinstance(bboxes[0], (np.ndarray, list, tuple)) and len(bboxes[0]) >= 4 : # Assuming bbox has at least 4 coordinates
                    faces_found_and_valid = True
                elif isinstance(bboxes[0], int) and bboxes[0] == 0 and len(bboxes) == 1:
                    faces_found_and_valid = False # Special signal for no faces from your detector
                    # print(f"ComfyUI-GazeNode: Frame {i}: No faces detected (detector returned [0]).")
                else:
                    faces_found_and_valid = False # Unrecognized bbox format
                    # print(f"ComfyUI-GazeNode: Frame {i}: Unrecognized bbox format: {bboxes[0]}")
            else:
                # bboxes is None or empty list
                faces_found_and_valid = False
                # print(f"ComfyUI-GazeNode: Frame {i}: No faces detected (bboxes is None or empty).")

            if faces_found_and_valid:
                # Ensure bboxes and lms5_all_faces are lists for consistent processing,
                # though your detector might already return them as lists of arrays.
                if not isinstance(bboxes, list): bboxes = list(bboxes)
                if not isinstance(lms5_all_faces, list): lms5_all_faces = list(lms5_all_faces)
                
                # Sort by area (width*height) to get the largest face
                # Assuming bbox format [x1, y1, x2, y2]
                try:
                    idxs_sorted = sorted(
                        range(len(bboxes)), 
                        key=lambda k: (bboxes[k][2] - bboxes[k][0]) * (bboxes[k][3] - bboxes[k][1]), 
                        reverse=True
                    )
                    if not idxs_sorted: # Should not happen if len(bboxes) > 0 from faces_found_and_valid
                         raise IndexError("Sorting bboxes resulted in empty list.")
                         
                    best_idx = idxs_sorted[0]
                    lms5_single_face = np.array(lms5_all_faces[best_idx])

                    # Gaze Prediction
                    current_gaze_dict = self.predictor(img_for_processing, lms5_single_face, undo_roll=True)

                    if current_gaze_dict:
                        # Smoothing
                        if smooth_predictions and prev_gaze_dict is not None:
                            current_gaze_dict = average_output_for_node(current_gaze_dict, prev_gaze_dict)
                        
                        prev_gaze_dict = current_gaze_dict.copy() # Update for next frame's smoothing

                        # Drawing
                        if enable_video:
                            temp_img_to_draw_on = processed_img_rgb_np # Draw on the RGB image
                        else:
                            temp_img_to_draw_on = np.zeros_like(processed_img_rgb_np) # Set to black

                        
                        if enable_eyes and current_gaze_dict.get('verts_eyes') is not None:
                            # Ensure draw_eyes from your utils.py expects RGB and returns RGB
                            temp_img_to_draw_on = draw_eyes(temp_img_to_draw_on, lms5_single_face, current_gaze_dict['verts_eyes'], draw_eyeball=show_eyeball,draw_iris=show_iris,draw_rings=False,squint_factor=squint_ratio,thickness_multiplier=line_thickness,color_mode=iris_mode)
                        
                        if enable_gaze_vector and current_gaze_dict.get('gaze_combined') is not None:
                            # Ensure draw_gaze from your utils.py expects RGB and returns RGB
                            temp_img_to_draw_on = draw_gaze(temp_img_to_draw_on, lms5_single_face, current_gaze_dict['gaze_combined'])
                        
                        processed_img_rgb_np = temp_img_to_draw_on
                    else: 
                        # Predictor returned no gaze_dict
                        prev_gaze_dict = None # Reset smoothing
                except IndexError as e_sort: # Catch errors from sorting/indexing if bboxes/lms5_all_faces are unexpectedly empty after checks
                    print(f"ComfyUI-GazeNode: Frame {i}: Error processing detected face (sorting/indexing): {e_sort}")
                    prev_gaze_dict = None
                except Exception as e_proc: # Catch any other error during prediction/drawing for this face
                    print(f"ComfyUI-GazeNode: Frame {i}: Error during prediction/drawing for detected face: {e_proc}")
                    prev_gaze_dict = None # Reset smoothing
            else: 
                # No valid faces detected in this frame
                prev_gaze_dict = None # Reset smoothing for the next frame

            batch_output_rgb_np.append(processed_img_rgb_np)
            pbar.update(1)
        
        # print(f"ComfyUI-GazeNode: Processing finished.") # Optional
        output_tensor = self.cv2_list_to_tensor(batch_output_rgb_np, images.device)


        # Offloading logic (remains similar, ensure GAZE_DETECTOR_INSTANCE and GAZE_PREDICTOR_INSTANCE are used)
        if offload_models_on_finish and GAZE_DETECTOR_INSTANCE is not None and GAZE_PREDICTOR_INSTANCE is not None:
            if CURRENT_GAZE_CONFIG.get("device") == "cuda":
                print("ComfyUI-GazeNode: Offloading gaze models to CPU...")
                try:
                    # InsightFace model might not have a .to('cpu') method.
                    # For PyTorch models:
                    if hasattr(GAZE_DETECTOR_INSTANCE, 'to') and callable(getattr(GAZE_DETECTOR_INSTANCE, 'to')):
                         # Check if it's not an InsightFace model directly, which handles its own device.
                         # This check might need refinement based on actual detector type.
                         if not "insightface" in str(type(GAZE_DETECTOR_INSTANCE.model)).lower(): # Heuristic
                            GAZE_DETECTOR_INSTANCE.to("cpu")
                         else:
                            print("ComfyUI-GazeNode: InsightFace detector manages its own device, offload not directly applicable via .to('cpu').")
                    if hasattr(GAZE_PREDICTOR_INSTANCE, 'to') and callable(getattr(GAZE_PREDICTOR_INSTANCE, 'to')):
                        GAZE_PREDICTOR_INSTANCE.to("cpu")
                    
                    CURRENT_GAZE_CONFIG["device"] = "cpu"
                    comfy.model_management.soft_empty_cache()
                    print("ComfyUI-GazeNode: Gaze models offload attempted.")
                except Exception as e:
                    print(f"ComfyUI-GazeNode: Error offloading models to CPU: {e}")
        return (output_tensor,)

# NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS (same as before)
NODE_CLASS_MAPPINGS = { "GazeNode": GazeNode }
NODE_DISPLAY_NAME_MAPPINGS = { "GazeNode": "Gaze Estimation 👁️" }