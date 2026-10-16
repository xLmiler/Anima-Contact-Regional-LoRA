import {localizeDOM} from "./i18n.js";
/* 实例区域涂改：在自动轮廓上绘制人物ID，不再依赖矩形。 */
const COLORS=["#55a9ff","#ffb05b","#72d9a1","#ec86b8","#a99bff","#f2d46d","#77d5de","#e89478","#d5ec86","#b775dc","#e76c64","#68b8b1","#c69f70","#8eabdb","#c8cfdf","#b2bf61"];
const el=(tag,text)=>{const e=document.createElement(tag);if(text)e.textContent=text;return e;};
export function createOwnershipEditor({value,onChange,onNamesChange=()=>{},onRequestWidth=()=>{}}){
 let state,undo=[],redo=[],current="char1",radius=14,opacity=.5,width=1024,height=1024,bg=null,labels=null,sourceId="",drawing=null,seedMode=false;
 try{state=JSON.parse(value);}catch{state={version:1,strokes:[]};}state.strokes||=[];
 let payload=null,cards=[],bindings=null,showInactive=false,updateVersion=0,destroyed=false;
 const root=el('div');Object.assign(root.style,{background:'#182130',border:'1px solid #405371',borderRadius:'8px',padding:'10px',color:'#d8e5f8',font:'12px/1.5 system-ui,"Microsoft YaHei","Noto Sans CJK SC"',width:'100%',maxWidth:'100%',minWidth:'0',height:'100%',overflowX:'hidden',overflowY:'auto',boxSizing:'border-box',contain:'inline-size'});
 const css=el('style');css.textContent='.anima-own *{box-sizing:border-box;min-width:0}.anima-own select{max-width:100%}.anima-own input{max-width:100%}.anima-own .role-card{display:flex;align-items:center;gap:7px;min-width:0;padding:6px;border:1px solid #49636a;border-radius:8px;background:#24383e;cursor:pointer}.anima-own .role-card.selected{border-color:#f0b72f;box-shadow:inset 0 0 0 1px #f0b72f}.anima-own .role-card input{width:100%;background:#1c2b32;border:0;color:#e9e7d3;padding:3px;border-radius:4px}';root.classList.add('anima-own');root.append(css);
 const title=el('div'),heading=el('span','归属编辑器 · 实例标签');title.append(heading);title.style.marginBottom='8px';root.append(title);
 const legend=el('div');legend.setAttribute('aria-label','角色颜色与名称对照');Object.assign(legend.style,{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(155px,1fr))',gap:'6px',marginBottom:'8px'});root.append(legend);
 const inactive=el('button','显示未绑定 / 已忽略人物');inactive.style.marginBottom='6px';inactive.hidden=true;root.append(inactive);inactive.onclick=()=>{showInactive=!showInactive;inactive.textContent=showInactive?'收起未绑定 / 已忽略人物':'显示未绑定 / 已忽略人物';renderOptions();renderLegend();};
 const widen=el('button','加宽节点');widen.onclick=()=>onRequestWidth(120);title.append(' ',widen);
 const row=el('div');Object.assign(row.style,{display:'flex',gap:'6px',alignItems:'center',flexWrap:'wrap'});root.append(row);
 const select=el('select');select.setAttribute('aria-label','画笔所属角色');select.dataset.preserveValues='true';select.style.maxWidth='230px';for(const id of ['erase']){const o=el('option',id==='erase'?'橡皮擦':id==='background'?'背景种子':id);o.value=id;select.append(o);}row.append(select);
 const brush=el('input');brush.type='range';brush.min=2;brush.max=80;brush.value=radius;brush.setAttribute('aria-label','画笔半径');brush.style.width='105px';const text=el('span',`${radius}px`);row.append(brush,text);
 const undoBtn=el('button','撤销'),redoBtn=el('button','重做'),clear=el('button','清空笔画');row.append(undoBtn,redoBtn,clear);
 for(const b of [select,undoBtn,redoBtn,clear])Object.assign(b.style,{background:'#2b3a51',color:'#e0ebff',border:'1px solid #536b8c',borderRadius:'4px',padding:'5px',cursor:'pointer'});
 const alpha=el('input');alpha.type='range';alpha.min=0;alpha.max=100;alpha.value=50;alpha.setAttribute('aria-label','身份颜色透明度');alpha.style.width='100px';const alphaRow=el('div');alphaRow.append(el('span','身份颜色透明度 '),alpha);alphaRow.style.margin='7px 0';root.append(alphaRow);
 const stage=el('div');Object.assign(stage.style,{width:'100%',aspectRatio:'1',maxHeight:'480px',position:'relative',border:'1px solid #425574',touchAction:'none'});const canvas=el('canvas');canvas.setAttribute('aria-label','人物归属绘制画布');Object.assign(canvas.style,{display:'block',width:'100%',height:'100%',cursor:'crosshair',touchAction:'none'});stage.append(canvas);root.append(stage);
 const status=el('div','先运行一次区域流程以载入参考图与自动人物轮廓；涂改后重新运行。');status.style.margin='8px 0';root.append(status);
 const transfer=el('div'),copyBtn=el('button','复制笔画'),pasteBtn=el('button','粘贴笔画');transfer.append(copyBtn,pasteBtn);root.append(transfer);
 copyBtn.onclick=async()=>{try{await navigator.clipboard.writeText(JSON.stringify(state));status.textContent='笔画已复制；可粘贴至同一参考图的种子节点。';}catch{status.textContent='浏览器未允许剪贴板；可复制整个种子节点到目标工作流。';}};
 pasteBtn.onclick=async()=>{try{const parsed=JSON.parse(await navigator.clipboard.readText());if(!Array.isArray(parsed.strokes)||parsed.strokes.length>5000)throw Error('无效笔画');if(parsed.strokes.length&&sourceId&&parsed.source_id!==sourceId)throw Error('参考图、宽高或人数不同，请先核对原图和设置');const before=JSON.stringify(state);state=parsed;history(before);paint();}catch(e){status.textContent='无法粘贴：'+e.message;}};
 const help=el('div','选择区域标识并绘制归属修正。笔画随工作流保存；接触边界在执行时从修正标签重建。');root.append(help);
 function notify(){state.source_id=sourceId||state.source_id||'';onChange(JSON.stringify(state));undoBtn.disabled=!undo.length;redoBtn.disabled=!redo.length;status.textContent=`已保存 ${state.strokes.length} 条笔画；${seedMode?"重新运行分割。":"重新运行后用于采样。"}`;}
 function visibleCards(){return cards.filter(c=>!bindings||showInactive||['active','duplicate'].includes(bindings[c.id]));}
 function renderOptions(){select.replaceChildren();for(const card of [...visibleCards(),...(seedMode?[{id:'background'}]:[]),{id:'erase'}]){const o=el('option',card.id==='erase'?'橡皮擦':card.id==='background'?'背景种子':card.id+' · '+(card.label||card.id));o.value=card.id;select.append(o);}if(![...select.options].some(o=>o.value===current))current=visibleCards()[0]?.id||'erase';select.value=current;}
 function renderLegend(){
  legend.replaceChildren();
  for(const card of visibleCards()){
   const box=el('div');box.className='role-card'+(card.id===current?' selected':'');box.dataset.charId=card.id;
   const swatch=el('span');Object.assign(swatch.style,{width:'13px',height:'42px',flex:'0 0 13px',borderRadius:'5px',background:card.color||COLORS[Number(card.id.slice(4))-1]});box.append(swatch);
   if(card.thumbnail){const im=el('img');im.src=card.thumbnail;Object.assign(im.style,{width:'38px',height:'48px',objectFit:'contain',flex:'0 0 38px'});box.append(im);}
   const info=el('div');info.style.flex='1';if(bindings&&bindings[card.id]!=='active'){const badge=el('small',({ignored:'已忽略 · 底模',unbound:'未绑定 · 底模',duplicate:'重复绑定，请检查'})[bindings[card.id]]||'');badge.style.color='#f3bc57';info.append(badge);}const id=el('strong',card.id+' · '+(card.color||COLORS[Number(card.id.slice(4))-1]));id.style.fontSize='10px';const name=el('input');name.maxLength=100;name.value=card.label||card.id;name.setAttribute('aria-label','角色名称 '+card.id);
   name.onchange=()=>{card.label=name.value.trim()||card.id;const option=[...select.options].find(x=>x.value===card.id);if(option)option.textContent=card.id+' · '+card.label;onNamesChange(Object.fromEntries(cards.map(c=>[c.id,c.label||c.id])));};name.onclick=e=>e.stopPropagation();info.append(id,name);box.append(info);box.onclick=()=>{current=card.id;select.value=current;renderLegend();paint();};legend.append(box);
  }
 }
 function history(before){if(before!==JSON.stringify(state)){undo.push(before);redo=[];notify();}}
 function paint(){
  const rect=stage.getBoundingClientRect();if(!rect.width||!rect.height)return;const dpr=Math.min(2,window.devicePixelRatio||1);canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  const c=canvas.getContext('2d');c.fillStyle='#1e2b3e';c.fillRect(0,0,canvas.width,canvas.height);if(bg)c.drawImage(bg,0,0,canvas.width,canvas.height);
  const overlay=document.createElement('canvas');overlay.width=canvas.width;overlay.height=canvas.height;const o=overlay.getContext('2d');
  if(labels){o.imageSmoothingEnabled=false;o.drawImage(labels,0,0,overlay.width,overlay.height);const data=o.getImageData(0,0,overlay.width,overlay.height);for(let i=0;i<data.data.length;i+=4)if(data.data[i]+data.data[i+1]+data.data[i+2]<20)data.data[i+3]=0;o.putImageData(data,0,0);}
  for(const s of [...state.strokes,...(drawing?[drawing]:[])]){
   if(!s.points?.length)continue;o.globalCompositeOperation=s.char_id==='erase'?'destination-out':'source-over';o.strokeStyle=s.char_id==='erase'?'#000':s.char_id==='background'?'#e626a6':COLORS[Number(s.char_id.slice(4))-1];o.fillStyle=o.strokeStyle;o.lineCap='round';o.lineJoin='round';o.lineWidth=2*s.radius*Math.min(overlay.width,overlay.height);o.beginPath();s.points.forEach(([x,y],i)=>i?o.lineTo(x*overlay.width,y*overlay.height):o.moveTo(x*overlay.width,y*overlay.height));o.stroke();const [x,y]=s.points[0];o.beginPath();o.arc(x*overlay.width,y*overlay.height,o.lineWidth/2,0,Math.PI*2);o.fill();
  }
  c.globalAlpha=opacity;c.drawImage(overlay,0,0);c.globalAlpha=1;
 }
 function point(e){const b=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-b.left)/b.width)),Math.max(0,Math.min(1,(e.clientY-b.top)/b.height))];}
 canvas.onpointerdown=e=>{if(e.button!==0||!labels)return;if(state.strokes.length&&state.source_id&&state.source_id!==sourceId){state={version:1,source_id:sourceId,strokes:[]};undo=[];redo=[];notify();}e.preventDefault();e.stopPropagation();drawing={char_id:current,radius:radius/Math.min(width,height),points:[point(e)]};canvas.setPointerCapture(e.pointerId);paint();};
 canvas.onpointermove=e=>{if(!drawing)return;e.preventDefault();drawing.points.push(point(e));paint();};
 canvas.onpointerup=e=>{if(!drawing)return;const before=JSON.stringify(state);state.strokes.push(drawing);drawing=null;history(before);if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);status.textContent=`已保存 ${state.strokes.length} 条笔画；${seedMode?"重新运行分割。":"重新运行后用于采样。"}`;paint();};
 canvas.onpointercancel=()=>{drawing=null;paint();};select.onchange=()=>{current=select.value;renderLegend();};
 brush.oninput=()=>{radius=Number(brush.value);text.textContent=radius+'px';};alpha.oninput=()=>{opacity=Number(alpha.value)/100;paint();};
 undoBtn.onclick=()=>{if(!undo.length)return;redo.push(JSON.stringify(state));state=JSON.parse(undo.pop());notify();paint();};
 redoBtn.onclick=()=>{if(!redo.length)return;undo.push(JSON.stringify(state));state=JSON.parse(redo.pop());notify();paint();};
 clear.onclick=()=>{const before=JSON.stringify(state);state={version:1,strokes:[],source_id:sourceId};history(before);notify();status.textContent='旧笔画已清空。重新运行可载入变化后的结构参考。';paint();};
 root.addEventListener('pointerdown',e=>e.stopPropagation());root.addEventListener('wheel',e=>e.stopPropagation(),{passive:true});
 const releaseLocale=localizeDOM(root);
 const observer=new ResizeObserver(paint);observer.observe(stage);
 return {element:root,sync(raw){try{state=JSON.parse(raw);state.strokes||=[];paint();}catch{}},setBindings(value){if(JSON.stringify(value)===JSON.stringify(bindings))return;bindings=value;inactive.hidden=!cards.some(c=>!['active','duplicate'].includes(bindings[c.id]));renderOptions();renderLegend();paint();},getPreviewData(){return payload?JSON.parse(JSON.stringify({...payload,characters:cards,view_state:{current,radius,opacity}})):null;},async update(data){
   const token=++updateVersion;if(destroyed)return;const reset=!!(state.strokes.length&&state.source_id&&data.source_id&&state.source_id!==data.source_id);if(reset){state={version:1,source_id:data.source_id,strokes:[]};undo=[];redo=[];onChange(JSON.stringify(state));}bg=null;labels=null;
   payload=JSON.parse(JSON.stringify(data));cards=data.characters.map(c=>({...c,color:c.color||COLORS[Number(c.id.slice(4))-1]}));
   seedMode=data.mode==='seeds';help.textContent=seedMode?'在可见人物内部标注前景种子，使用背景种子排除背景。橡皮仅删除标注。笔画仅适用于相同参考图、尺寸与身份数量。':'选择区域标识并绘制归属修正。笔画随工作流保存；接触边界在执行时从修正标签重建。';heading.textContent=seedMode?'身份种子 · 前景与背景标注':'归属编辑器 · 实例标签';width=data.width;height=data.height;sourceId=data.source_id;brush.max=Math.min(80,Math.floor(Math.min(width,height)*.2));radius=Math.min(radius,Number(brush.max));brush.value=radius;text.textContent=radius+'px';stage.style.aspectRatio=`${width}/${height}`;stage.style.maxWidth=`${480*width/height}px`;stage.style.marginInline='auto';if(data.view_state){current=data.view_state.current||current;radius=Number(data.view_state.radius)||radius;opacity=Math.max(0,Math.min(1,data.view_state.opacity??opacity));brush.value=radius;text.textContent=radius+'px';alpha.value=opacity*100;}renderOptions();renderLegend();
   const load=src=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=src;});const loaded=await Promise.all([load(data.background),load(data.labels)]);if(destroyed||token!==updateVersion)return;[bg,labels]=loaded;status.textContent=seedMode?`${width}×${height}；参考图已载入。标注身份与背景种子后重新运行分割。`:`${width}×${height}；已载入全部 ${data.characters.length} 人，颜色与角色见上方图例。预览缓存仅供编辑，涂改后重新运行。`;if(reset||data.reset_reason)status.textContent='参考区域变化，已自动清空旧笔画。'+status.textContent;paint();
 },getData(){return JSON.parse(JSON.stringify(state));},destroy(){releaseLocale();destroyed=true;updateVersion++;observer.disconnect();}};
}
