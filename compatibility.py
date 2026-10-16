"""旧工作流的可执行接口别名；菜单仍按功能提供规范节点。"""
from __future__ import annotations
import copy,json
from pathlib import Path
from . import nodes
from .model_store import available,ensure_model

CONTRACTS=json.loads(Path(__file__).with_name('compatibility_schema.json').read_text())

def input_types(cls):
    spec=copy.deepcopy(CONTRACTS[cls.__name__]['inputs'])
    fields={k:v for group in spec.values() for k,v in group.items()}
    if 'sam2_checkpoint' in fields:fields['sam2_checkpoint'][0]=available('sam2',['sam2.1_hiera_tiny.pt'])
    if 'detector' in fields:fields['detector'][0]=available('anima_contact_detectors',['person_detect_v1.3_s.onnx','person_detect_v1.1_m.onnx'])
    if 'model_name' in fields:
        known=['animaTileRepair_v20.safetensors'] if cls.__name__=='AnimaContactTileModelV8' else ['anima-lllite-any-test-like-v2.safetensors','anima-lllite-regional-exp-v3.safetensors']
        fields['model_name'][0]=list(dict.fromkeys(known+available('model_patches')+available('controlnet')))
    return spec

def execute(self,**kwargs):
    contract=CONTRACTS[type(self).__name__];target=contract['target']
    kwargs.pop('unmatched_policy',None)
    if target=='AnimaPersonSegment':
        foreground=kwargs.pop('foreground_model','disabled')
        if foreground!='disabled':
            if foreground not in ('auto','isnetis.onnx'):raise ValueError('Use the independent foreground node for a custom foreground model.')
            kwargs['foreground_mask']=nodes.AnimaForeground().segment(kwargs['image'],'isnet')['result'][0]
    if target=='AnimaLLLiteLoader':
        from .shape_control import load_lllite
        name=kwargs['model_name'];presets={'animaTileRepair_v20.safetensors':'tile','anima-lllite-any-test-like-v2.safetensors':'outline','anima-lllite-regional-exp-v3.safetensors':'regional'}
        if name in presets:result=nodes.AnimaLLLiteLoader().load(presets[name])
        else:
            import folder_paths
            path=folder_paths.get_full_path('model_patches',name) or folder_paths.get_full_path_or_raise('controlnet',name)
            result=(load_lllite(path,kwargs.get('profile','any-test-v2')),f'Validated model: {name}')
    else:
        instance=nodes.NODE_CLASS_MAPPINGS[target]();result=getattr(instance,instance.FUNCTION)(**kwargs)
    # 旧涂改接口只有四个输出；不改变其槽位。
    count=len(contract['outputs'])
    if isinstance(result,dict):result=dict(result,result=tuple(result['result'][:count]))
    else:result=tuple(result[:count])
    return result

NODE_CLASS_MAPPINGS={}
NODE_DISPLAY_NAME_MAPPINGS={}
for name,contract in CONTRACTS.items():
    target=nodes.NODE_CLASS_MAPPINGS[contract['target']]
    cls=type(name,(),dict(INPUT_TYPES=classmethod(input_types),RETURN_TYPES=tuple(contract['outputs']),RETURN_NAMES=target.RETURN_NAMES[:len(contract['outputs'])],FUNCTION='execute',CATEGORY=target.CATEGORY,DESCRIPTION=target.DESCRIPTION,DEPRECATED=True,execute=execute))
    NODE_CLASS_MAPPINGS[name]=cls
    NODE_DISPLAY_NAME_MAPPINGS[name]=nodes.NODE_DISPLAY_NAME_MAPPINGS[contract['target']]
