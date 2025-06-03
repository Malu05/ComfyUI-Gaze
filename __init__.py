# ComfyUI/custom_nodes/ComfyUI-Gaze/__init__.py

try:
    # This import will fail if GazeNode.py itself has syntax errors OR
    # if GazeNode.py re-raises an ImportError due to its own failed internal imports.
    from .GazeNode import GazeNode 

    NODE_CLASS_MAPPINGS = {
        "GazeNode": GazeNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "GazeNode": "Gaze Estimation 👁️"
    }

    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    
    print("✅ ComfyUI-Gaze: Custom 'GazeNode' loaded successfully.")

except ImportError as e: # Catches if GazeNode.py itself can't be imported or re-raises
    error_message = f"❌ ComfyUI-Gaze: Failed to import 'GazeNode' or its critical dependencies.\n"
    error_message += f"   Error: {e}\n"
    error_message += f"   This could be due to GazeNode.py not being found, syntax errors in GazeNode.py, "
    error_message += f"or GazeNode.py failing to import its own required modules (e.g., from the './gaze_models/models/' structure).\n"
    error_message += f"   Please check the GazeNode.py file and ensure all its internal imports are correct and dependencies are met.\n"
    if hasattr(e, '__traceback__') and e.__traceback__:
        # This line number will be from GazeNode.py if it re-raised, or __init__.py if GazeNode.py not found
        error_message += f"   Error likely occurred near line: {e.__traceback__.tb_lineno}\n"
    print(error_message)

    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
except Exception as e: # Catch any other unexpected errors
    error_message = f"❌ ComfyUI-Gaze: An unexpected error occurred while trying to load 'GazeNode'.\n"
    error_message += f"   Error: {e}\n"
    if hasattr(e, '__traceback__') and e.__traceback__:
        error_message += f"   Error occurred at line: {e.__traceback__.tb_lineno}\n"
    print(error_message)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}