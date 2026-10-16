"""Model locations, verified downloads and startup directory initialization."""
from __future__ import annotations
import hashlib,logging,os,threading,urllib.request,uuid
from pathlib import Path
MODELS={
 'isnet':('anima_contact_segmentation/isnetis.onnx','https://huggingface.co/skytnt/anime-seg/resolve/493cb60893f47441b26ec4fb9a306bce9e342982/isnetis.onnx','f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99'),
 'toonout':('anima_contact_segmentation/birefnet-toonout-fp16.onnx','https://huggingface.co/sprited/birefnet-toonout-onnx/resolve/2ded6fe6063fe146b7b30d8b4cfc5636df322525/birefnet-toonout-fp16.onnx','213a8a98ee426ef8f02d247eb5a5a9889359e37c2e1e7e31e282d61034d08a83'),
 'detector-small':('anima_contact_detectors/person_detect_v1.3_s.onnx','https://huggingface.co/deepghs/anime_person_detection/resolve/e39c744c22432ad01f91dd254fe2b02c8d878b8c/person_detect_v1.3_s/model.onnx','6da88929438cd442e31e45ff4f934dd2d7eb9cf7a423c22885d650ee52550f90'),
 'detector-medium':('anima_contact_detectors/person_detect_v1.1_m.onnx','https://huggingface.co/deepghs/anime_person_detection/resolve/e39c744c22432ad01f91dd254fe2b02c8d878b8c/person_detect_v1.1_m/model.onnx','c80365a98ca8388fd2f01fd5abea0439e92265bceceb048bef38b0849303467c'),
 'sam2-tiny':('sam2/sam2.1_hiera_tiny.pt','https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/de431c4043854a71d8101e17995dfe596bf101a5/sam2.1_hiera_tiny.pt','7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'),
 'outline':('model_patches/anima-lllite-any-test-like-v2.safetensors','https://huggingface.co/kohya-ss/Anima-LLLite/resolve/36ba7f2f498a1ca63cc77fc7a1298652f72d4524/anima-lllite-any-test-like-v2.safetensors','8863424f04a2445815a827b22cbda557a085d2752e05fab54a2b303821a9e3b2'),
 'tile':('controlnet/animaTileRepair_v20.safetensors','https://huggingface.co/LeFeujitif/sandbox/resolve/00729a0ce2caf54a10c3268979d2bc5ecda7d1b8/animaTileRepair_v20.safetensors','be5a9ead6527329237a66a0eb234ccea4cddb140c1d1e1b7d0776e43f5ff1330'),
 'regional':('model_patches/anima-lllite-regional-exp-v3.safetensors','https://huggingface.co/Sen-sou/Anima-LLLite-Regional-Controlnet/resolve/f1b7beeddb4175d2d20b14e10538db62bbb441a6/anima-lllite-regional-exp-v3.safetensors','f8cdc9897a4eaa50b0f9bc18dda098fb096eeb64df001589492b9774d0cf5cc5')}
FOLDERS={'anima_contact_detectors':{'.onnx'},'anima_contact_segmentation':{'.onnx'},'sam2':{'.pt'},'model_patches':{'.safetensors'},'controlnet':{'.safetensors'},'loras':{'.safetensors'},'sam3':{'.safetensors'}}
_LOCK=threading.RLock();_VALID={}
def initialize_directories():
 import folder_paths
 root=Path(folder_paths.models_dir)
 for name,extensions in FOLDERS.items():
  paths,exts=folder_paths.folder_names_and_paths.get(name,([str(root/name)],set()))
  for path in paths:Path(path).mkdir(parents=True,exist_ok=True)
  folder_paths.folder_names_and_paths[name]=(list(paths),set(exts)|extensions)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def ensure_model(key,auto_download=True):
 if key=='sam3':return ensure_sam3(auto_download)
 import folder_paths
 relative,url,sha=MODELS[key];folder,name=relative.split('/',1)
 initialize_directories()
 existing=folder_paths.get_full_path(folder,name) if hasattr(folder_paths,'get_full_path') else None
 path=Path(existing) if existing else Path(folder_paths.folder_names_and_paths[folder][0][0])/name
 with _LOCK:
  if path.is_file():
   st=path.stat();signature=(str(path),st.st_size,st.st_mtime_ns)
   if _VALID.get(signature)!=sha:
    if digest(path)!=sha:raise RuntimeError(f'Model checksum mismatch: {path}. Existing files are not overwritten.')
    _VALID[signature]=sha
   return str(path)
  if not auto_download:raise FileNotFoundError(f'Model not found: {path}. Enable automatic download or install the model manually.')
  part=path.with_name(path.name+'.'+uuid.uuid4().hex+'.part')
  try:
   logging.info('[AnimaContact] Downloading %s to %s',key,path)
   with urllib.request.urlopen(url,timeout=60) as response,part.open('wb') as f:
    for chunk in iter(lambda:response.read(1024*1024),b''):f.write(chunk)
   if digest(part)!=sha:raise RuntimeError(f'Download checksum mismatch: {key}')
   os.replace(part,path)
   st=path.stat();_VALID[(str(path),st.st_size,st.st_mtime_ns)]=sha
   return str(path)
  finally:part.unlink(missing_ok=True)
def available(folder,known=()):
 import folder_paths
 return list(dict.fromkeys([*known,*folder_paths.get_filename_list(folder)]))

def verify_sam3_file(path,expected):
 """校验固定修订的 LFS 权重或 Git 原始配置文件。"""
 path=Path(path)
 if not path.is_file():return False
 st=path.stat();signature=(str(path),st.st_size,st.st_mtime_ns);value=next(iter(expected.values()))
 if _VALID.get(signature)==value:return True
 if 'sha256' in expected:actual=digest(path)
 else:
  h=hashlib.sha1();h.update(f'blob {st.st_size}\0'.encode());h.update(path.read_bytes());actual=h.hexdigest()
 if actual!=value:return False
 _VALID[signature]=value;return True

def ensure_sam3(auto_download=True):
 """从公开镜像下载与上游校验一致的 Transformers 文件；无需令牌。"""
 import folder_paths,shutil
 initialize_directories();path=Path(folder_paths.folder_names_and_paths['sam3'][0][0])/'facebook-sam3';path.mkdir(parents=True,exist_ok=True)
 with _LOCK:
  for name,expected in SAM3_FILES.items():
   target=path/name
   if target.exists():
    if not verify_sam3_file(target,expected):raise RuntimeError(f'SAM3 checksum mismatch: {target}. Existing files are not overwritten.')
    continue
   if not auto_download:raise FileNotFoundError(f'SAM3 file missing: {target}. Enable automatic download or install the complete official Transformers model directory.')
   try:
    from huggingface_hub import hf_hub_download
   except ImportError as e:raise RuntimeError('SAM3 requires requirements_sam3.txt installed with the ComfyUI Python environment.') from e
   try:cached=hf_hub_download(repo_id=SAM3_REPO,revision=SAM3_REVISION,filename=name,token=False)
   except Exception as e:
    code=getattr(getattr(e,'response',None),'status_code',None)
    if code in (401,403) or type(e).__name__=='GatedRepoError':
     raise RuntimeError('The public SAM3 download source is currently unavailable. Check access to https://huggingface.co/bodhicitta/sam3 or install the complete verified model directory manually. Authentication is not required for this source.') from None
    raise RuntimeError(f'SAM3 download failed for {name}: {type(e).__name__}. Check network access or install the complete model directory manually.') from None
   if not verify_sam3_file(cached,expected):raise RuntimeError(f'SAM3 downloaded file failed verification: {name}')
   part=target.with_name(target.name+'.'+uuid.uuid4().hex+'.part')
   try:shutil.copyfile(cached,part);os.replace(part,target)
   finally:part.unlink(missing_ok=True)
 return str(path)

SAM3_REPO="bodhicitta/sam3"
SAM3_REVISION='cba430d22f6fdc3f06ad3841274ec7bb55885f2f'
SAM3_FILES={'config.json': {'git_blob': '3e600e765b0f1b4c52680aed9ad242c9a90f7ed1'}, 'merges.txt': {'git_blob': '76e821f1b6f0a9709293c3b6b51ed90980b3166b'}, 'model.safetensors': {'sha256': '6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a'}, 'processor_config.json': {'git_blob': '9d3bd6df8a54852cd56087a3ae019cce051de2b2'}, 'special_tokens_map.json': {'git_blob': 'cf0682d6de72c1547f41b4f6d7c59f62deffef94'}, 'tokenizer.json': {'git_blob': 'bd1e367764006ad6a2249b1dcc0b20879c7c211f'}, 'tokenizer_config.json': {'git_blob': '1f20c0274c5d4581b5ccdcbdddd7a22729b5c57b'}, 'vocab.json': {'git_blob': '182766ce89b439768edadda342519f33802f5364'}}
