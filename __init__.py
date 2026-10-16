"""Anima regional conditioning, ownership masks and structural control."""
from .model_store import initialize_directories
try:initialize_directories()
except ModuleNotFoundError as error:
 if error.name!='folder_paths':raise
from .nodes import NODE_CLASS_MAPPINGS,NODE_DISPLAY_NAME_MAPPINGS
from .compatibility import NODE_CLASS_MAPPINGS as COMPAT_NODES,NODE_DISPLAY_NAME_MAPPINGS as COMPAT_NAMES
NODE_CLASS_MAPPINGS={**NODE_CLASS_MAPPINGS,**COMPAT_NODES}
NODE_DISPLAY_NAME_MAPPINGS={**NODE_DISPLAY_NAME_MAPPINGS,**COMPAT_NAMES}
WEB_DIRECTORY='./web'
__all__=['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS','WEB_DIRECTORY']
