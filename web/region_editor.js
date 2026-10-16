import {localizeDOM,translate} from "./i18n.js";
/* 中文区域编辑面板。只维护归一化 JSON；不依赖第三方前端库。 */
const COLORS=["#55a9ff","#ffb05b","#72d9a1","#ec86b8","#a99bff","#f2d46d","#77d5de","#e89478","#d5ec86","#b775dc","#e76c64","#68b8b1","#c69f70","#8eabdb","#c8cfdf","#b2bf61"];
const KINDS={body:"角色区域",face:"面部重点",ownership:"肢体归属",contact:"交融区域"};
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
const clone=x=>JSON.parse(JSON.stringify(x));
function valid(data){
  if(!data||!Array.isArray(data.characters)||!data.characters.length||data.characters.length>16||!Array.isArray(data.regions))throw Error("需要 1~16 个角色和 regions 数组");
  const ids=new Set(data.characters.map(c=>c.id));
  if(ids.size!==data.characters.length||[...ids].some(id=>!/^char(?:[1-9]|1[0-6])$/.test(id)))throw Error("角色 ID 必须为唯一的 char1~char16");
  const rids=new Set();
  for(const r of data.regions){
    if(!r.id||rids.has(r.id)||!(r.kind in KINDS))throw Error("区域 ID 或类型错误");
    rids.add(r.id);
    if(r.kind!=="contact"&&!ids.has(r.char_id))throw Error("区域的角色不存在");
    if(![r.x,r.y,r.w,r.h].every(Number.isFinite)||r.w<=0||r.h<=0||r.x<0||r.y<0||r.x+r.w>1.000001||r.y+r.h>1.000001)throw Error("区域超出画布或大小为零");
  }
  if(data.regions.length>128)throw Error("最多 128 个选框");
  return clone(data);
}
function element(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text)e.textContent=text;return e;}

export function createRegionEditor({value,onChange,width=1024,height=1024}){
  let data,selected=null,charId="char1",kind="body",mode="select",drag=null,background=null;
  let undo=[],redo=[];let syncing=false;
  const root=element("div","acv5");root.tabIndex=0;
  root.innerHTML=`<style>
  .acv5{font:12px/1.45 system-ui,"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",sans-serif;color:#e9edf7;background:#171d2a;border:1px solid #3a465c;border-radius:9px;padding:10px;box-sizing:border-box;width:100%;height:100%;min-width:330px;overflow:auto}
  .acv5 *{box-sizing:border-box}.acv5 button,.acv5 select{font:inherit;color:#e9edf7;background:#293449;border:1px solid #465873;border-radius:5px;padding:5px 7px;cursor:pointer;min-height:29px}
  .acv5 button:hover{background:#384b68}.acv5 button.active{background:#345984;border-color:#79b7fc}.acv5 button:disabled{opacity:.4;cursor:default}
  .acv5 .row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:7px}.acv5 .grow{flex:1;min-width:65px}.acv5 .muted{color:#a5b4ca;font-size:11px}
  .acv5 .stage{position:relative;width:100%;background:#1d2636;border:1px solid #465873;overflow:hidden;border-radius:5px;touch-action:none}
  .acv5 canvas{width:100%;height:100%;display:block;touch-action:none;cursor:crosshair}.acv5 .status{padding-top:7px;min-height:39px;white-space:pre-line;color:#b6cbe7}
  .acv5 .error{color:#ffaaaa}.acv5 details{margin-top:5px;color:#aabbd1}.acv5 summary{cursor:pointer}.acv5 .badge{color:#8fc7ff;font-weight:600}
  </style>`;
  const head=element("div","row");const title=element("span","badge","角色区域 · 可拖拽");const size=element("span","muted grow");
  head.append(title,size);root.append(head);
  const toolbar=element("div","row");const chars=element("select");chars.setAttribute("aria-label","当前角色");
  const add=element("button","","＋角色");add.title="建立下一个 char，随后在画布拖出角色框";
  const removeChar=element("button","","删角色");
  const kinds=element("select");kinds.setAttribute("aria-label","区域类型");
  for(const [k,v] of Object.entries(KINDS)){const o=element("option","",v);o.value=k;kinds.append(o);}
  toolbar.append(chars,add,removeChar,kinds);root.append(toolbar);
  const actions=element("div","row");const drawBtn=element("button","","画框"),selectBtn=element("button","active","选择 / 移动");
  const deleteBtn=element("button","","删选框"),undoBtn=element("button","","撤销"),redoBtn=element("button","","重做");
  actions.append(drawBtn,selectBtn,deleteBtn,undoBtn,redoBtn);root.append(actions);
  const regions=element("select","grow");regions.setAttribute("aria-label","选框列表");
  const regrow=element("div","row");regrow.append(regions);root.append(regrow);
  const stage=element("div","stage"),canvas=element("canvas");canvas.setAttribute("aria-label","区域编辑画布");stage.append(canvas);root.append(stage);
  const status=element("div","status");status.setAttribute("role","status");root.append(status);
  const tools=element("div","row");const bgBtn=element("button","","参考图"),clearBg=element("button","","移除参考"),exp=element("button","","导出布局"),imp=element("button","","导入布局");
  tools.append(bgBtn,clearBg,exp,imp);root.append(tools);
  const help=element("details");const summary=element("summary","","操作与颜色说明");const helpText=element("div","muted",
    "区域坐标归一化并随工作流保存。主要区域、局部重点、显式归属和接触边界分别定义空间覆盖、局部优先级、归属分配和适配器释放。拖动可编辑位置与大小，Delete 删除，Ctrl/Cmd+Z 撤销。参考图仅用于浏览器对齐，不参与采样。");
  help.append(summary,helpText);root.append(help);
  const file=element("input");file.type="file";file.accept="image/*";file.hidden=true;root.append(file);
  const jsonFile=element("input");jsonFile.type="file";jsonFile.accept="application/json,.json";jsonFile.hidden=true;root.append(jsonFile);

  function snapshot(){return JSON.stringify(data);}
  function notify(){if(!syncing)onChange(snapshot());}
  function history(before){if(before!==snapshot()){undo.push(before);if(undo.length>160)undo.shift();redo=[];notify();}refresh();}
  function edit(fn){const before=snapshot();fn();history(before);}
  function color(r){return r.kind==="contact"?"#d591ff":COLORS[Number(r.char_id.slice(4))-1];}
  function refresh(){
    chars.replaceChildren();for(const c of data.characters){const o=element("option","",c.id);o.value=c.id;chars.append(o);}chars.value=charId;
    kinds.value=kind;drawBtn.classList.toggle("active",mode==="draw");selectBtn.classList.toggle("active",mode==="select");
    regions.replaceChildren();const empty=element("option","","选择选框（或点击画布）");empty.value="";regions.append(empty);
    for(const r of data.regions){const o=element("option","",`${r.kind==="contact"?"交融":r.char_id} · ${KINDS[r.kind]} · ${Math.round(r.w*width)}×${Math.round(r.h*height)}`);o.value=r.id;regions.append(o);}regions.value=selected||"";
    add.disabled=data.characters.length>=16;removeChar.disabled=data.characters.length<=1;deleteBtn.disabled=!selected;undoBtn.disabled=!undo.length;redoBtn.disabled=!redo.length;
    size.textContent=`${width} × ${height} · ${data.characters.length} 角色 / ${data.regions.length} 框`;
    stage.style.aspectRatio=`${width} / ${height}`;
    // 竖幅画布不无限拉高节点；坐标仍按完整画布换算。
    stage.style.maxWidth=`${Math.min(100000,440*width/height)}px`;
    stage.style.marginInline="auto";
    const r=data.regions.find(r=>r.id===selected);
    status.classList.remove("error");status.textContent=r?`${r.kind==="contact"?"交融区":r.char_id+" · "+KINDS[r.kind]}：左 ${Math.round(r.x*width)} / 上 ${Math.round(r.y*height)} / 宽 ${Math.round(r.w*width)} / 高 ${Math.round(r.h*height)} px\n拖动框内移动；拖控制点调整大小。`:
      `${charId} · ${KINDS[kind]}：${mode==="draw"?"在画布按下并拖出新选框。":"点击选框选择；也可从列表选择被遮住的框。"}`;
    paint();
  }
  function handles(r){return [[r.x,r.y,"nw"],[r.x+r.w/2,r.y,"n"],[r.x+r.w,r.y,"ne"],[r.x,r.y+r.h/2,"w"],[r.x+r.w,r.y+r.h/2,"e"],[r.x,r.y+r.h,"sw"],[r.x+r.w/2,r.y+r.h,"s"],[r.x+r.w,r.y+r.h,"se"]];}
  function paint(){
    const bounds=stage.getBoundingClientRect();if(!bounds.width||!bounds.height)return;
    const dpr=Math.min(2,window.devicePixelRatio||1),cw=Math.round(bounds.width*dpr),ch=Math.round(bounds.height*dpr);
    if(canvas.width!==cw||canvas.height!==ch){canvas.width=cw;canvas.height=ch;}
    const ctx=canvas.getContext("2d");ctx.clearRect(0,0,cw,ch);ctx.fillStyle="#1d2636";ctx.fillRect(0,0,cw,ch);
    if(background){ctx.globalAlpha=.65;ctx.drawImage(background,0,0,cw,ch);ctx.globalAlpha=1;}
    ctx.strokeStyle="#ffffff12";ctx.lineWidth=dpr;
    for(let i=1;i<16;i++){ctx.beginPath();ctx.moveTo(i*cw/16,0);ctx.lineTo(i*cw/16,ch);ctx.moveTo(0,i*ch/16);ctx.lineTo(cw,i*ch/16);ctx.stroke();}
    for(const r of data.regions){
      const c=color(r),x=r.x*cw,y=r.y*ch,rw=r.w*cw,rh=r.h*ch;ctx.fillStyle=c+"22";ctx.fillRect(x,y,rw,rh);
      ctx.strokeStyle=c;ctx.lineWidth=(r.id===selected?3:1.5)*dpr;ctx.setLineDash(r.kind==="face"?[5*dpr,3*dpr]:r.kind==="contact"?[9*dpr,4*dpr]:[]);ctx.strokeRect(x,y,rw,rh);ctx.setLineDash([]);
      const label=translate(`${r.kind==="contact"?"交融":r.char_id} ${KINDS[r.kind]}`);ctx.font=`${11*dpr}px system-ui,"Microsoft YaHei","PingFang SC","Noto Sans CJK SC"`;const tw=ctx.measureText(label).width;
      const labelY=r.kind==="face"?Math.max(y,y+rh-18*dpr):y;
      ctx.fillStyle="#111b2be8";ctx.fillRect(x+1,labelY+1,Math.min(tw+8*dpr,cw-x),17*dpr);ctx.fillStyle=c;ctx.fillText(label,x+4*dpr,labelY+13*dpr);
    }
    const selectedRect=data.regions.find(r=>r.id===selected);
    if(selectedRect){ctx.fillStyle="#ffffff";ctx.strokeStyle="#152135";ctx.lineWidth=dpr;for(const [hx,hy] of handles(selectedRect)){ctx.fillRect(hx*cw-4*dpr,hy*ch-4*dpr,8*dpr,8*dpr);ctx.strokeRect(hx*cw-4*dpr,hy*ch-4*dpr,8*dpr,8*dpr);}}
  }
  function point(e){const r=canvas.getBoundingClientRect();return {x:clamp((e.clientX-r.left)/r.width,0,1),y:clamp((e.clientY-r.top)/r.height,0,1)};}
  function setSelection(r){selected=r?.id||null;if(r&&r.kind!=="contact")charId=r.char_id;if(r)kind=r.kind;}
  canvas.addEventListener("pointerdown",e=>{
    if(e.button!==0)return;e.preventDefault();e.stopPropagation();root.focus({preventScroll:true});
    const p=point(e),before=snapshot();const r=data.regions.find(r=>r.id===selected);
    const b=canvas.getBoundingClientRect();const hitHandle=mode==="select"&&r?handles(r).find(([x,y])=>Math.abs(p.x-x)*b.width<=9&&Math.abs(p.y-y)*b.height<=9):null;
    if(hitHandle){drag={type:"resize",handle:hitHandle[2],id:r.id,start:p,original:clone(r),before};}
    else if(mode==="draw"){
      if(data.regions.length>=128){status.textContent="最多 128 个选框";return;}
      const nr={id:`r-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,7)}`,kind,char_id:kind==="contact"?"":charId,x:p.x,y:p.y,w:.001,h:.001};data.regions.push(nr);selected=nr.id;
      drag={type:"draw",id:nr.id,start:p,before};
    }else{
      const target=[...data.regions].reverse().find(r=>p.x>=r.x&&p.x<=r.x+r.w&&p.y>=r.y&&p.y<=r.y+r.h);
      setSelection(target);
      if(target)drag={type:"move",id:target.id,start:p,original:clone(target),before};
    }
    if(drag)canvas.setPointerCapture(e.pointerId);refresh();
  });
  canvas.addEventListener("pointermove",e=>{
    if(!drag)return;e.preventDefault();e.stopPropagation();const p=point(e),r=data.regions.find(r=>r.id===drag.id);if(!r)return;
    const dx=p.x-drag.start.x,dy=p.y-drag.start.y,o=drag.original;
    if(drag.type==="draw"){r.x=Math.min(p.x,drag.start.x);r.y=Math.min(p.y,drag.start.y);r.w=Math.abs(dx);r.h=Math.abs(dy);}
    else if(drag.type==="move"){r.x=clamp(o.x+dx,0,1-o.w);r.y=clamp(o.y+dy,0,1-o.h);}
    else{
      let x0=o.x,y0=o.y,x1=o.x+o.w,y1=o.y+o.h;const h=drag.handle;
      if(h.includes("w"))x0=clamp(o.x+dx,0,x1-.005);if(h.includes("e"))x1=clamp(o.x+o.w+dx,x0+.005,1);
      if(h.includes("n"))y0=clamp(o.y+dy,0,y1-.005);if(h.includes("s"))y1=clamp(o.y+o.h+dy,y0+.005,1);
      Object.assign(r,{x:x0,y:y0,w:x1-x0,h:y1-y0});
    }
    refresh();
  });
  function end(e){
    if(!drag)return;e.stopPropagation();const before=drag.before;const r=data.regions.find(r=>r.id===drag.id);
    if(r&&(r.w<.004||r.h<.004)){data.regions=data.regions.filter(x=>x.id!==r.id);selected=null;}
    drag=null;if(canvas.hasPointerCapture?.(e.pointerId))canvas.releasePointerCapture(e.pointerId);history(before);
  }
  canvas.addEventListener("pointerup",end);canvas.addEventListener("pointercancel",e=>{if(drag){data=JSON.parse(drag.before);drag=null;selected=null;refresh();}});
  chars.addEventListener("change",()=>{charId=chars.value;selected=null;refresh();});
  kinds.addEventListener("change",()=>{kind=kinds.value;selected=null;refresh();});
  drawBtn.onclick=()=>{mode="draw";selected=null;refresh();};selectBtn.onclick=()=>{mode="select";refresh();};
  add.onclick=()=>edit(()=>{const id=Array.from({length:16},(_,i)=>`char${i+1}`).find(id=>!data.characters.some(c=>c.id===id));if(!id)return;data.characters.push({id,label:id});charId=id;kind="body";mode="draw";selected=null;});
  removeChar.onclick=()=>{if(data.characters.length<=1)return;edit(()=>{data.characters=data.characters.filter(c=>c.id!==charId);data.regions=data.regions.filter(r=>r.char_id!==charId);charId=data.characters[0].id;selected=null;});};
  function remove(){if(selected)edit(()=>{data.regions=data.regions.filter(r=>r.id!==selected);selected=null;});}
  deleteBtn.onclick=remove;
  undoBtn.onclick=()=>{if(!undo.length)return;redo.push(snapshot());data=JSON.parse(undo.pop());selected=null;charId=data.characters.some(c=>c.id===charId)?charId:data.characters[0].id;notify();refresh();};
  redoBtn.onclick=()=>{if(!redo.length)return;undo.push(snapshot());data=JSON.parse(redo.pop());selected=null;charId=data.characters.some(c=>c.id===charId)?charId:data.characters[0].id;notify();refresh();};
  regions.onchange=()=>{setSelection(data.regions.find(r=>r.id===regions.value));mode="select";refresh();};
  root.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA"||e.target.tagName==="SELECT")return;if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"){e.preventDefault();e.stopPropagation();(e.shiftKey?redoBtn:undoBtn).click();}else if(e.key==="Delete"||e.key==="Backspace"){e.preventDefault();e.stopPropagation();remove();}});
  bgBtn.onclick=()=>file.click();clearBg.onclick=()=>{background=null;paint();};
  file.onchange=async()=>{const f=file.files[0];if(!f)return;const url=URL.createObjectURL(f);const img=new Image();img.onload=()=>{background=img;paint();URL.revokeObjectURL(url);};img.onerror=()=>{status.textContent="无法读取参考图";URL.revokeObjectURL(url);};img.src=url;file.value="";};
  exp.onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:"application/json"}));const a=element("a");a.href=url;a.download="anima_regions.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  imp.onclick=()=>jsonFile.click();jsonFile.onchange=async()=>{try{const next=valid(JSON.parse(await jsonFile.files[0].text()));edit(()=>{data=next;charId=data.characters[0].id;selected=null;});}catch(e){status.classList.add("error");status.textContent="导入失败："+e.message;}jsonFile.value="";};
  function sync(raw){try{syncing=true;data=valid(typeof raw==="string"?JSON.parse(raw):raw);if(!data.characters.some(c=>c.id===charId))charId=data.characters[0].id;selected=null;refresh();}catch(e){status.classList.add("error");status.textContent="布局数据错误："+e.message;}finally{syncing=false;}}
  try{data=valid(typeof value==="string"?JSON.parse(value):value);}catch(e){data={version:1,characters:[{id:"char1",label:"char1"}],regions:[]};status.textContent=e.message;}
  charId=data.characters[0].id;
  // 画布内的鼠标操作不冒泡给 LiteGraph，防止拖框变成拖整个节点。
  root.addEventListener("pointerdown",e=>e.stopPropagation());root.addEventListener("wheel",e=>e.stopPropagation(),{passive:true});
  const observer=new ResizeObserver(()=>paint());observer.observe(stage);refresh();
  const releaseLocale=localizeDOM(root,paint);
  return {element:root,sync,setSize(w,h){width=w;height=h;refresh();},destroy(){releaseLocale();observer.disconnect();},getData:()=>clone(data)};
}
