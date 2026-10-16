"""Migrate supported version-suffixed workflows without modifying the source file."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCHEMA=json.loads((ROOT/'node_schema.json').read_text())
ALIASES={'AnimaContactCanvasV5':'AnimaRegionCanvas','AnimaContactSceneV5':'AnimaSceneConditioning','AnimaContactSceneV9':'AnimaSceneConditioning','AnimaContactRoleV8':'AnimaIdentity','AnimaContactRoleV9':'AnimaIdentity','AnimaContactApplyV8':'AnimaRegionalApply','AnimaContactApplyV9':'AnimaRegionalApply','AnimaContactAnimeDetectV7':'AnimaPersonDetect','AnimaContactAnimeRegionsV7':'AnimaPersonSegment','AnimaContactRegionsPaintV8':'AnimaOwnershipEditor','AnimaContactRegionsPaintV9':'AnimaOwnershipEditor','AnimaContactMaskImageV7':'AnimaIdentityMapImport','AnimaContactReferenceSeedsV61':'AnimaIdentitySeeds','AnimaContactSeedRegionsV7':'AnimaSeedSegment','AnimaContactLayoutControlV7':'AnimaLayoutControl','AnimaContactLLLiteModelV7':'AnimaLLLiteLoader','AnimaContactTileModelV8':'AnimaLLLiteLoader','AnimaContactLLLiteApplyV7':'AnimaContactLLLiteApply','AnimaContactTileHiresV8':'AnimaTileUpscale','AnimaContactInspectV5':'AnimaRegionalInspect'}
ORDER={'AnimaContactRoleV9':['char_id','identity_prompt','edge_softness','identity_gain','self_q_out_gain','self_kv_gain','cross_gain','mlp_gain','lora_stack_json','unmatched_policy'],'AnimaContactRoleV8':['char_id','lora_name','lora_strength','identity_prompt','edge_softness','lora_mode','identity_gain','self_q_out_gain','self_kv_gain','cross_gain','mlp_gain'],'AnimaContactRegionsPaintV9':['strokes_json','island_pixels','hole_pixels','smooth_radius','identity_names_json'],'AnimaContactRegionsPaintV8':['strokes_json','island_pixels','hole_pixels','smooth_radius'],'AnimaContactSceneV9':['scene_prompt','quality_prompt','negative_prompt'],'AnimaContactSceneV5':['scene_prompt','negative_prompt'],'AnimaContactLLLiteModelV7':['model_name','profile'],'AnimaContactTileModelV8':['model_name'],'AnimaContactAnimeRegionsV7':['sam2_checkpoint','canvas_width','canvas_height','analysis_resolution','contact_width','foreground_model','refinement']}
TYPES={'ANIMA_CONTACT_LAYOUT_V5':'ANIMA_LAYOUT','ANIMA_CONTACT_ROLES_V5':'ANIMA_IDENTITIES','ANIMA_CONTACT_PIPE_V5':'ANIMA_PIPELINE','ANIMA_CONTACT_SEEDS_V61':'ANIMA_SEEDS','ANIMA_ANIME_BOXES_V7':'ANIMA_PEOPLE','ANIMA_SHAPE_LLLITE_V7':'ANIMA_LLLITE','ANIMA_SCENE_POLICY_V9':'ANIMA_SCENE_POLICY'}
def defaults(kind):
 vals={};ins=[]
 for group in SCHEMA[kind]['inputs'].values():
  for key,e in group.items():
   t,o=e
   if isinstance(t,list) or t in ['INT','FLOAT','STRING','BOOLEAN'] and not o.get('forceInput'):vals[key]=copy.deepcopy(o.get('default',t[0] if isinstance(t,list) else 0))
   else:ins.append(dict(name=key,type=t,link=None))
 return vals,ins

def migrate(data):
 data=copy.deepcopy(data);old_nodes={n['id']:copy.deepcopy(n) for n in data['nodes']};fg_nodes=[]
 for node in data['nodes']:
  if node['type']=='Model采样与诊断AuraFlow':
   node['type']='ModelSamplingAuraFlow';node.setdefault('properties',{})['Node name for S&R']=node['type']
  old=node['type'];kind=ALIASES.get(old)
  if old=='AnimaLLLiteApply' and any(x['name']=='pipeline' for x in node.get('inputs',[])):kind='AnimaContactLLLiteApply'
  if not kind:
   if old.startswith('AnimaContact'):raise ValueError(f'Retired node requires manual replacement: {old}')
   continue
  values,inputs=defaults(kind);props=node.setdefault('properties',{});saved=props.get('animaWidgetState',{}).get('values') or node.get('widgets_values_named')
  if not saved:
   order=ORDER.get(old)
   if order is None:order=list(values)
   saved=dict(zip(order,node.get('widgets_values',[])))
  saved=dict(saved)
  if kind=='AnimaIdentity' and 'lora_stack_json' not in saved:
   name=saved.get('lora_name');enabled=saved.get('lora_mode')!='disabled' and name and name!='None';saved['lora_stack_json']=json.dumps({'version':1,'entries':[dict(name=name,strength=saved.get('lora_strength',1),enabled=True,purpose='identity')] if enabled else []})
  if kind=='AnimaLLLiteLoader':
   name=saved.get('model_name','');saved['model_name']='tile' if old=='AnimaContactTileModelV8' or 'TileRepair' in name else 'regional' if 'regional' in name else 'outline' if 'any-test' in name else name
   if saved['model_name'] not in ['outline','regional','tile']:raise ValueError(f'Custom control requires manual preset selection: {name}')
  if kind=='AnimaPersonSegment' and saved.get('foreground_model','disabled')!='disabled':fg_nodes.append(node['id'])
  for key in values:
   if key in saved:values[key]=saved[key]
  node['type']=kind;node.pop('title',None);node['widgets_values']=list(values.values());node['widgets_values_named']=values;props['Node name for S&R']=kind;props['animaWidgetState']={'version':1,'type':kind,'values':copy.deepcopy(values)};node['inputs']=inputs;node['outputs']=[dict(name=k,type=t,links=[],slot_index=i) for i,(k,t) in enumerate(zip(SCHEMA[kind]['output_names'],SCHEMA[kind]['outputs']))]
 by={n['id']:n for n in data['nodes']};links=[]
 for e in data['links']:
  lid,src,slot,dst,index,typ=e;target=by[dst];old=old_nodes[dst];name=old['inputs'][index]['name'];idx=next((i for i,x in enumerate(target['inputs']) if x['name']==name),None)
  if idx is None:
   if target['type'] not in SCHEMA:idx=index
   else:
    vals=target['widgets_values_named']
    if name not in vals:raise ValueError(f'Input no longer supported: {target["type"]}.{name}')
    idx=len(target['inputs']);target['inputs'].append(dict(name=name,type=TYPES.get(typ,typ),widget={'name':name},link=None))
  target['inputs'][idx]['link']=lid;links.append([lid,src,slot,dst,idx,TYPES.get(typ,typ)])
  if by[src]['type'] in SCHEMA:by[src]['outputs'][slot]['links'].append(lid)
 last=max(by,default=0);linkid=max((e[0] for e in links),default=0)
 for targetid in fg_nodes:
  target=by[targetid];image=next(e for e in links if e[3]==targetid and target['inputs'][e[4]]['name']=='image');last+=1;vals,ins=defaults('AnimaForeground');new=dict(id=last,type='AnimaForeground',pos=[target['pos'][0]-650,target['pos'][1]+500],size=[520,350],flags={},mode=0,order=0,inputs=ins,outputs=[dict(name=k,type=t,links=[],slot_index=i) for i,(k,t) in enumerate(zip(SCHEMA['AnimaForeground']['output_names'],SCHEMA['AnimaForeground']['outputs']))],properties={'Node name for S&R':'AnimaForeground','animaWidgetState':dict(version=1,type='AnimaForeground',values=copy.deepcopy(vals))},widgets_values=list(vals.values()),widgets_values_named=vals);data['nodes'].append(new)
  linkid+=1;new['inputs'][0]['link']=linkid;links.append([linkid,image[1],image[2],last,0,'IMAGE']);by[image[1]]['outputs'][image[2]].setdefault('links',[]).append(linkid)
  idx=next(i for i,x in enumerate(target['inputs']) if x['name']=='foreground_mask');linkid+=1;target['inputs'][idx]['link']=linkid;new['outputs'][0]['links'].append(linkid);links.append([linkid,last,0,targetid,idx,'MASK'])
 data['links']=links;data['last_node_id']=last;data['last_link_id']=linkid
 return data

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('input',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
 if args.input.resolve()==args.output.resolve():parser.error('Choose a separate output path to preserve the original workflow.')
 data=migrate(json.loads(args.input.read_text(encoding='utf-8')));args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');print(args.output)
if __name__=='__main__':main()
