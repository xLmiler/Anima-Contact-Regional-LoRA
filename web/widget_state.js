import {canonicalType} from "./compatibility.js";
/* 按输入名保存，隔离普通控件、隐藏STRING和只用于显示的DOM控件。 */
const ORDER={
 AnimaIdentity:['char_id','identity_prompt','edge_softness','identity_gain','self_q_out_gain','self_kv_gain','cross_gain','mlp_gain','lora_stack_json'],
 AnimaOwnershipEditor:['strokes_json','island_pixels','hole_pixels','smooth_radius','identity_names_json'],
};
const clone=v=>v===undefined?null:JSON.parse(JSON.stringify(v));
export function displayOnly(widget){widget.serialize=false;widget.options||={};widget.options.serialize=false;return widget;}
export function preserveInputs(node){
 const widgets=new Map(node.widgets.filter(w=>w.name).map(w=>[w.name,w]));
 const order=(ORDER[canonicalType(node)]||[...widgets.keys()]).filter(n=>widgets.has(n));
 const defaults=Object.fromEntries(order.map(n=>[n,clone(widgets.get(n).value)]));
 const configure=node.onConfigure;
 node.onConfigure=function(info={},...args){
  const out=configure?.call(this,info,...args);
  const saved=info.properties?.animaWidgetState;
  const values={...(saved?.type===this.comfyClass?saved.values:info.widgets_values_named||Object.fromEntries(order.map((n,i)=>[n,info.widgets_values?.[i]])))};
  if(canonicalType(this)==='AnimaIdentity'&&values.char_id!==undefined&&!/^char([1-9]|1[0-6])$/.test(values.char_id)){
   const id=info.properties?.animaIdentityCard?.id||info.title?.match(/char(?:1[0-6]|[1-9])\b/)?.[0];
   if(id)values.char_id=id;
   const candidates=Object.values(values).filter(v=>typeof v==='string');
   const stack=candidates.find(v=>{try{return Array.isArray(JSON.parse(v)?.entries);}catch{return false;}});
   const prompt=candidates.filter(v=>!v.trim().startsWith('{')&&v.includes(',')).sort((a,b)=>b.length-a.length)[0];
   if(stack)values.lora_stack_json=stack;if(prompt)values.identity_prompt=prompt;
   console.warn('[AnimaContact] 旧文件身份参数曾错位，已按保存的色签/JSON恢复可识别字段；请核对数值。');
  }
  for(const name of order){let v=values?.[name];if(v===undefined)continue;
   if(typeof defaults[name]==='number'&&(typeof v!=='number'||!Number.isFinite(v)))v=defaults[name];
   widgets.get(name).value=clone(v);
  }
  return out;
 };
 const serialize=node.onSerialize;
 node.onSerialize=function(data){serialize?.call(this,data);const values=Object.fromEntries(order.map(n=>[n,clone(widgets.get(n).value)]));
  this.properties||={};this.properties.animaWidgetState={version:1,type:this.comfyClass,values};
  data.properties||={};data.properties.animaWidgetState=clone(this.properties.animaWidgetState);
  data.widgets_values=order.map(n=>values[n]);data.widgets_values_named=values;
 };
 return widgets;
}
