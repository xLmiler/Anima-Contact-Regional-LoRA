"""只读取safetensors头部，不加载权重，不上传文件。用于诊断适配器键名与模型架构的匹配情况。"""
import argparse,json,struct,collections,re
from pathlib import Path
def inspect(path):
    with Path(path).open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        if length>64*1024*1024:raise ValueError('异常safetensors头部大小')
        header=json.loads(f.read(length))
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    patterns={'anima_blocks':r'(?:^|[._])blocks[._](\d+)[._](?:self_attn|cross_attn|mlp)',
              'sd_unet':r'(?:down_blocks|up_blocks|input_blocks|output_blocks)',
              'transformer_blocks':r'transformer_blocks',
              'llm_adapter':r'llm_adapter'}
    hits={name:sum(bool(re.search(pattern,k)) for k in tensors) for name,pattern in patterns.items()}
    return {'file_name':Path(path).name,'tensor_count':len(tensors),'metadata':header.get('__metadata__',{}),'name_pattern_counts':hits,
            'prefix_counts':dict(collections.Counter(k.split('.')[0][:80] for k in tensors).most_common(15)),
            'sample_keys_and_shapes':[{'key':k,'shape':v.get('shape')} for k,v in list(tensors.items())[:20]],
            'note':'只有键名统计，不足以保证架构兼容；本插件按真实投影形状匹配，不会跨架构强行加载。'}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('file');ap.add_argument('--out',type=Path);args=ap.parse_args();text=json.dumps(inspect(args.file),ensure_ascii=False,indent=2)
    if args.out:args.out.write_text(text,encoding='utf-8')
    print(text)
if __name__=='__main__':main()
