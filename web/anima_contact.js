import {containWidget} from "./dom_bounds.js";
import {localizeDOM} from "./i18n.js";
import {canonicalType,repairControlIdentifier} from "./compatibility.js";
import {app} from "../../scripts/app.js";
import {createRegionEditor} from "./region_editor.js";
import {createOwnershipEditor} from "./ownership_editor.js";
import {createLoraGroup} from "./lora_group.js";
import {displayOnly,preserveInputs} from "./widget_state.js";
import {trackNode,publishCatalog,refreshGraph} from "./identity_sync.js";

function hidden(node,name){const w=node.widgets.find(w=>w.name===name);if(w){w.hidden=true;w.computeSize=()=>[0,-4];if(w.inputEl)w.inputEl.style.display='none';}return w;}
function resizeNode(node,width,height){const size=[Math.max(580,width||node.size?.[0]||580),Math.max(height||0,node.size?.[1]||0)];if(node.setSize)node.setSize(size);else node.size=size;node.setDirtyCanvas?.(true,true);}
let loraChoices;
async function getLoras(){if(!loraChoices)loraChoices=(async()=>{const {api}=await import('../../scripts/api.js');const response=await api.fetchApi('/models/loras');const d=await response.json();return d;})();return loraChoices.catch(e=>{loraChoices=null;throw e;});}


app.registerExtension({
  name:"AnimaContact.Interface",
  beforeConfigureGraph(data){repairControlIdentifier(data);},
  afterConfigureGraph(){refreshGraph(app.graph);},
  beforeQueuePrompt(){refreshGraph(app.graph);},
  nodeCreated(node){
    trackNode(node);
    const kind=canonicalType(node);
    if(["AnimaIdentity","AnimaOwnershipEditor"].includes(kind))preserveInputs(node);
    if(kind==='AnimaIdentity'){
      const raw=hidden(node,'lora_stack_json');if(!raw)return;let panelHeight=250;node.properties||={};
      const panel=createLoraGroup({value:raw.value,onChange(v){raw.value=v;node.graph?.change?.();},onResize(h){panelHeight=h;resizeNode(node,node.size[0],Math.max(720,h+440));}});
      const updateIdentity=()=>{const id=node.widgets.find(w=>w.name==='char_id')?.value;const link=(node.graph?.links?.get?.(node.inputs?.find(x=>x.name==='layout')?.link)??node.graph?.links?.[node.inputs?.find(x=>x.name==='layout')?.link]);const parent=node.graph?.getNodeById?.(link?.origin_id);const card=parent?.properties?.animaOwnershipPayload?.characters?.find(x=>x.id===id);if(card){node.properties.animaIdentityCard=card;panel.setIdentity(card);}else if(node.properties.animaIdentityCard?.id===id)panel.setIdentity(node.properties.animaIdentityCard);else panel.setIdentity({id,label:id,color:'#78a8b7'});};
      const idWidget=node.widgets.find(w=>w.name==='char_id');if(idWidget){const callback=idWidget.callback;idWidget.callback=function(...args){const out=callback?.apply(this,args);updateIdentity();return out;};}
      const widget=node.addDOMWidget('identity_lora_group','ANIMA_LORA_GROUP',panel.element,{serialize:false,hideOnZoom:false,getMinHeight:()=>panelHeight,getMaxHeight:()=>panelHeight});displayOnly(widget);containWidget(node,widget,{height:()=>panelHeight});
      getLoras().then(x=>panel.setChoices(x)).catch(e=>console.warn('[AnimaContact] LoRA列表不可用，仍保留已存条目',e));
      const executed=node.onExecuted;node.onExecuted=function(message){executed?.call(this,message);const c=message?.anima_identity?.at(-1);if(c){this.properties.animaIdentityCard=c;panel.setIdentity(c);}};
      const configured=node.onConfigure;node.onConfigure=function(...args){const out=configured?.apply(this,args);queueMicrotask(()=>{panel.sync(raw.value);updateIdentity();});return out;};
      if(node.properties.animaIdentityCard)panel.setIdentity(node.properties.animaIdentityCard);
      const removed=node.onRemoved;node.onRemoved=function(...a){panel.destroy();return removed?.apply(this,a);};resizeNode(node,node.size?.[0],Math.max(720,panelHeight+440));node.animaLoraGroup=panel;return;
    }
    if(["AnimaOwnershipEditor","AnimaIdentitySeeds"].includes(kind)){
      const raw=node.widgets.find(w=>w.name==='strokes_json');if(!raw)return;
      raw.hidden=true;raw.computeSize=()=>[0,-4];if(raw.inputEl)raw.inputEl.style.display='none';
      node.properties||={};const names=hidden(node,'identity_names_json');
      const panel=createOwnershipEditor({value:raw.value,onChange(v){raw.value=v;node.setDirtyCanvas?.(true,true);node.graph?.change?.();},onNamesChange(value){if(names)names.value=JSON.stringify(value);node.properties.animaOwnershipPayload=panel.getPreviewData();publishCatalog(node,node.properties.animaOwnershipPayload);node.graph?.change?.();},onRequestWidth(delta){resizeNode(node,(node.size?.[0]||580)+delta);}});
      const widget=node.addDOMWidget('ownership_panel','ANIMA_OWNERSHIP_EDITOR',panel.element,{serialize:false,hideOnZoom:false,getMinHeight:()=>820,getMaxHeight:()=>1100});displayOnly(widget);containWidget(node,widget,{height:820,minHeight:820,maxHeight:1100});
      let disposed=false;const load=async data=>{if(disposed||!data?.characters?.length)return;node.properties.animaOwnershipPayload=JSON.parse(JSON.stringify(data));await panel.update(data);if(!disposed)publishCatalog(node,data);};
      const executed=node.onExecuted;node.onExecuted=function(message){executed?.call(this,message);const data=message?.anima_mask_editor?.at(-1);if(data&&!disposed){if(data.strokes_state)raw.value=JSON.stringify(data.strokes_state);if(names&&data.names_state)names.value=JSON.stringify(data.names_state);panel.sync(raw.value);load(data).catch(console.error);}};
      const configured=node.onConfigure;node.onConfigure=function(...args){const out=configured?.apply(this,args);queueMicrotask(()=>{panel.sync(raw.value);const data=this.properties?.animaOwnershipPayload;if(data)load(data).catch(console.error);});return out;};
      const serialized=node.onSerialize;node.onSerialize=function(data){const out=serialized?.call(this,data);const payload=panel.getPreviewData();if(payload){this.properties.animaOwnershipPayload=payload;if(data){data.properties||={};data.properties.animaOwnershipPayload=payload;}}return out;};
      const onResize=node.onResize;let resizing=false;node.onResize=function(...args){const out=onResize?.apply(this,args);panel.element.style.maxWidth=Math.max(240,(this.size?.[0]||580)-20)+'px';if(!resizing&&(this.size?.[0]||0)<580){resizing=true;resizeNode(this,580);resizing=false;}return out;};
      if(node.properties.animaOwnershipPayload)load(node.properties.animaOwnershipPayload).catch(console.error);
      const removed=node.onRemoved;node.onRemoved=function(...args){disposed=true;panel.destroy();return removed?.apply(this,args);};
      resizeNode(node,node.size?.[0],1050);node.animaOwnershipPanel=panel;return;
    }
    if(["AnimaRegionalInspect","AnimaPersonDetect","AnimaRegionalApply","AnimaForeground","AnimaModelDownload","AnimaContactSAM3Segment","AnimaContactForegroundOwnership"].includes(kind)){
      const text=document.createElement("pre");text.textContent="采样完成后显示本阶段运行诊断。";
      Object.assign(text.style,{height:"100%",maxHeight:"100%",whiteSpace:"pre-wrap",overflowWrap:"anywhere",wordBreak:"break-word",overflow:"auto",font:"12px/1.5 system-ui",padding:"10px",color:"#c9ddf7",background:"#1a2435",borderRadius:"6px",margin:"0"});
      const report=node.addDOMWidget("runtime_report","ANIMA_RUNTIME_REPORT",text,{serialize:false});displayOnly(report);containWidget(node,report,{height:170});const releaseLocale=localizeDOM(text);const removed=node.onRemoved;node.onRemoved=function(...args){releaseLocale();return removed?.apply(this,args);};
      const executed=node.onExecuted;node.onExecuted=function(message){executed?.call(this,message);text.textContent=(message?.text||[]).join("\n");};
      return;
    }
    if(kind!=="AnimaRegionCanvas")return;
    const find=name=>node.widgets.find(w=>w.name===name);
    const raw=find("layout_json"),width=find("canvas_width"),height=find("canvas_height");
    if(!raw||!width||!height)return;
    // 数据仍通过标准 STRING widget 序列化和发送，只隐藏其数字/JSON 编辑外观。
    raw.type="converted-widget";raw.computeSize=()=>[0,-4];
    if(raw.inputEl)raw.inputEl.style.display="none";
    const panel=createRegionEditor({value:raw.value,width:width.value,height:height.value,onChange(value){
      raw.value=value;raw.callback?.(value);node.setDirtyCanvas?.(true,true);node.graph?.change?.();
    }});
    const widget=node.addDOMWidget("region_panel","ANIMA_REGION_EDITOR",panel.element,{serialize:false,hideOnZoom:false});
    displayOnly(widget);containWidget(node,widget,{height:640,minHeight:640,maxHeight:900});
    const initialConfigure=node.onConfigure;
    node.onConfigure=function(...args){const out=initialConfigure?.apply(this,args);queueMicrotask(()=>{panel.sync(raw.value);panel.setSize(width.value,height.value);});return out;};
    for(const w of [width,height]){const old=w.callback;w.callback=function(...args){const out=old?.apply(this,args);panel.setSize(width.value,height.value);return out;};}
    const removed=node.onRemoved;node.onRemoved=function(...args){panel.destroy();return removed?.apply(this,args);};
    // 只设初始宽度，不在 redraw/执行结束后反复重置用户手动拉宽的尺寸。
    resizeNode(node,node.size?.[0],800);
    node.animaRegionPanel=panel;
  }
});
