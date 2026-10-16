"""Download and verify catalogue models into an explicit ComfyUI model directory."""
import argparse,importlib.util,sys,types
from pathlib import Path

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--models-root',required=True,type=Path);parser.add_argument('--model',required=True);args=parser.parse_args()
 fp=types.ModuleType('folder_paths');fp.models_dir=str(args.models_root.resolve());fp.folder_names_and_paths={};fp.get_full_path=lambda f,n:str(args.models_root/f/n) if (args.models_root/f/n).is_file() else None;sys.modules['folder_paths']=fp
 spec=importlib.util.spec_from_file_location('anima_model_store',Path(__file__).resolve().parents[1]/'model_store.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if args.model not in [*m.MODELS,'sam3']:parser.error('Model must be one of: '+', '.join(sorted([*m.MODELS,'sam3'])))
 print(m.ensure_model(args.model,True))
if __name__=='__main__':main()
