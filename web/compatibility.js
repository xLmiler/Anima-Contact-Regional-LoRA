/* Stable functional interfaces for saved workflows. */
export const ALIASES={
  "AnimaContactAnimeDetectV7": "AnimaPersonDetect",
  "AnimaContactAnimeRegionsV7": "AnimaPersonSegment",
  "AnimaContactApplyV9": "AnimaRegionalApply",
  "AnimaContactInspectV5": "AnimaRegionalInspect",
  "AnimaContactLLLiteApplyV7": "AnimaContactLLLiteApply",
  "AnimaContactLLLiteModelV7": "AnimaLLLiteLoader",
  "AnimaContactLayoutControlV7": "AnimaLayoutControl",
  "AnimaContactRegionsPaintV9": "AnimaOwnershipEditor",
  "AnimaContactRoleV9": "AnimaIdentity",
  "AnimaContactSceneV9": "AnimaSceneConditioning",
  "AnimaContactTileHiresV8": "AnimaTileUpscale",
  "AnimaContactTileModelV8": "AnimaLLLiteLoader"
};
export const canonicalType=node=>ALIASES[node?.comfyClass||node?.type]||(node?.comfyClass||node?.type);
export function repairControlIdentifier(data){
 for(const node of data?.nodes||[]){
  if(node.type==='Model采样与诊断AuraFlow'){
   node.type='ModelSamplingAuraFlow';
   node.properties||={};node.properties['Node name for S&R']=node.type;
   continue;
  }
  // 仅处理区域管线接口；原生 model/model_patch 节点必须保持原生含义。
  if(node.type!=='AnimaLLLiteApply'||!node.inputs?.some(x=>x.name==='pipeline'))continue;
  node.type='AnimaContactLLLiteApply';
  if(node.properties){node.properties['Node name for S&R']=node.type;if(node.properties.animaWidgetState)node.properties.animaWidgetState.type=node.type;}
 }
}
