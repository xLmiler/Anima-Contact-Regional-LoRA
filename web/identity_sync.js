import {canonicalType} from "./compatibility.js";
/* char ID + 实际连线决定身份，不按标题或LoRA文件名猜测。 */
const tracked=new Set();const history=new WeakMap();const muted=new Map();let timer;
export const edge=(g,id)=>g?.links?.get?.(id)??g?.links?.[id];
const nodes=g=>g?._nodes||[];
const input=(n,name)=>n.inputs?.findIndex(x=>x.name===name);
const source=(n,name)=>{const e=edge(n.graph,n.inputs?.[input(n,name)]?.link);return e?{node:n.graph?.getNodeById?.(e.origin_id),slot:e.origin_slot}:null;};
const role=n=>canonicalType(n)==='AnimaIdentity';
function snapshot(n){if(!n.graph)return;const outputs=(n.outputs?.[0]?.links||[]).map(id=>edge(n.graph,id)).filter(Boolean).map(e=>({node:n.graph.getNodeById(e.target_id),slot:e.target_slot})).filter(x=>x.node&&(role(x.node)||canonicalType(x.node)==='AnimaRegionalApply'));history.set(n,{graph:n.graph,previous:source(n,'previous_roles'),outputs});}
function connect(previous,target,slot){if(previous?.node)previous.node.connect?.(previous.slot,target,slot);else target.disconnectInput?.(slot);}
function effective(previous){const seen=new Set();while(previous?.node?.mode===2&&!seen.has(previous.node)){seen.add(previous.node);previous=(muted.get(previous.node)||history.get(previous.node))?.previous;}return previous;}
function matches(target,slot,previous){const e=edge(target.graph,target.inputs?.[slot]?.link);return previous?.node?e?.origin_id===previous.node.id&&e?.origin_slot===previous.slot:!e;}
function bypassMuted(n){
 if(n.mode===2){
  let s=muted.get(n);const first=!s;if(!s){const old=history.get(n);if(!old)return;s={...old,bypassed:old.previous};muted.set(n,s);}
  const previous=effective(s.previous);
  for(const out of s.outputs)if(out.node.graph===s.graph&&(first||matches(out.node,out.slot,s.bypassed))&&!matches(out.node,out.slot,previous))connect(previous,out.node,out.slot);
  s.bypassed=previous;
 }else if(muted.has(n)){
  const s=muted.get(n);muted.delete(n);
  if(n.graph!==s.graph||s.graph.getNodeById?.(n.id)!==n)return;
  for(const out of s.outputs)if(out.node.graph===s.graph&&matches(out.node,out.slot,s.bypassed))n.connect?.(0,out.node,out.slot);
 }
}

function usedRoles(g){
 const applies=nodes(g).filter(n=>canonicalType(n)==='AnimaRegionalApply'&&n.mode!==2&&n.mode!==4);if(!applies.length)return nodes(g).filter(n=>role(n)&&n.mode!==2&&n.mode!==4);
 const found=new Set(),visited=new Set();function visit(n){if(!n||visited.has(n))return;visited.add(n);if(role(n)){if(n.mode!==2&&n.mode!==4)found.add(n);visit(source(n,'previous_roles')?.node);}else if(n.type==='Reroute'||n.comfyClass==='Reroute')visit(source(n,n.inputs?.[0]?.name)?.node);}
 for(const a of applies)visit(source(a,'roles')?.node);return [...found];
}
export function refreshGraph(g){
 if(!g)return;const all=nodes(g);for(const n of all.filter(role))bypassMuted(n);
 const used=usedRoles(g);
 for(const n of all){
  if(n.animaOwnershipPanel){const payload=n.properties?.animaOwnershipPayload;if(!payload?.characters)continue;const same=all.filter(r=>role(r)&&source(r,'layout')?.node===n);
   const states=Object.fromEntries(payload.characters.map(c=>{const bound=same.filter(r=>r.widgets?.find(w=>w.name==='char_id')?.value===c.id);const active=bound.filter(r=>used.includes(r));return [c.id,active.length>1?'duplicate':active.length?'active':bound.some(r=>r.mode===2||r.mode===4)?'ignored':'unbound'];}));
   n.animaOwnershipPanel.setBindings?.(states);
  }
  if(role(n)){const parent=source(n,'layout')?.node;const id=n.widgets?.find(w=>w.name==='char_id')?.value;const c=parent?.properties?.animaOwnershipPayload?.characters?.find(c=>c.id===id);if(c){n.properties||={};n.properties.animaIdentityCard={...c};n.animaLoraGroup?.setIdentity(c);}}
 }
 for(const n of all.filter(role))if(n.mode!==2)snapshot(n);
}
export function publishCatalog(n,payload){if(!payload)return;refreshGraph(n.graph);}
export function trackNode(n){
 if(!['AnimaIdentity','AnimaOwnershipEditor','AnimaRegionalApply'].includes(canonicalType(n)))return;
 tracked.add(n);const changes=n.onConnectionsChange;n.onConnectionsChange=function(...a){const out=changes?.apply(this,a);queueMicrotask(()=>refreshGraph(this.graph));return out;};
 const removed=n.onRemoved;n.onRemoved=function(...a){const s=history.get(this);tracked.delete(this);muted.delete(this);const out=removed?.apply(this,a);
  if(role(this)&&s)queueMicrotask(()=>{for(const target of s.outputs){if(s.graph.getNodeById?.(target.node.id)!==target.node)continue;if(s.previous?.node&&s.graph.getNodeById?.(s.previous.node.id)!==s.previous.node)continue;if(target.node.inputs?.[target.slot]?.link==null)connect(effective(s.previous),target.node,target.slot);}refreshGraph(s.graph);});return out;};
 if(!timer){const tick=()=>{for(const g of new Set([...tracked].map(n=>n.graph).filter(Boolean)))refreshGraph(g);timer=tracked.size?setTimeout(tick,250):null;};timer=setTimeout(tick,0);}
}
