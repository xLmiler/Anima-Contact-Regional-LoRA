import {app} from '../../scripts/app.js';
const PAIRS={
 '角色区域 · 可拖拽':'Spatial Region Canvas', '当前角色':'Current Identity','＋角色':'+ Identity','删角色':'Remove Identity','区域类型':'Region Type','选择 / 移动':'Select / Move','删选框':'Delete Region','选框列表':'Region List','区域编辑画布':'Region Canvas','参考图':'Reference Image','移除参考':'Remove Reference','导出布局':'Export Layout','导入布局':'Import Layout','操作与颜色说明':'Editing and Color Legend','选择选框（或点击画布）':'Select a Region or Click the Canvas','建立下一个 char，随后在画布拖出角色框':'Create an identity identifier, then draw its ownership region',
 '区域坐标归一化并随工作流保存。主要区域、局部重点、显式归属和接触边界分别定义空间覆盖、局部优先级、归属分配和适配器释放。拖动可编辑位置与大小，Delete 删除，Ctrl/Cmd+Z 撤销。参考图仅用于浏览器对齐，不参与采样。':'Region coordinates are normalized and saved with the workflow. Body, focus, ownership and contact regions define spatial coverage, local priority, ownership assignment and adapter release. Drag to edit position and size; Delete removes and Ctrl/Cmd+Z undoes. Reference images are used only for alignment.',
 ' 角色 / ':' identities / ',' 框':' regions','交融区':'Contact Region','交融':'Contact','：左 ':': left ',' / 上 ':' / top ',' / 宽 ':' / width ',' / 高 ':' / height ',
 '拖动框内移动；拖控制点缩放。':'Drag inside to move; drag handles to resize.',

 '选择区域标识并绘制归属修正。笔画随工作流保存；接触边界在执行时从修正标签重建。':'Select a region identifier and paint ownership corrections. Strokes persist with the workflow; contact boundaries are rebuilt from corrected labels.',
 '人物归属绘制画布':'Ownership Painting Canvas','需要 1~16 个角色和 regions 数组':'Expected 1–16 identities and a regions array','角色 ID 必须为唯一的 char1~char16':'Identity identifiers must be unique within char1–char16','区域 ID 或类型错误':'Invalid region identifier or type','区域的角色不存在':'The region refers to an unknown identity','区域超出画布或大小为零':'Region bounds must be nonzero and inside the canvas','最多 128 个选框':'At most 128 regions are supported','无法读取参考图':'Cannot read reference image','拖动框内移动；拖控制点调整大小。':'Drag inside to move; drag handles to resize.','在画布按下并拖出新选框。':'Drag on the canvas to create a region.','点击选框选择；也可从列表选择被遮住的框。':'Click to select a region or use the list for occluded regions.',
 '区域画布':'Region Canvas','角色区域':'Body Region','面部重点':'Face Region','肢体归属':'Ownership Region','交融区域':'Contact Region','选取/移动':'Select / Move','画框':'Draw Region','添加角色':'Add Identity','删除角色':'Remove Identity','删除选框':'Delete Region','导出JSON':'Export JSON','导入JSON':'Import JSON','载入参考图':'Load Reference','清除底图':'Clear Reference','选择已有区域':'Select Region','选区':'Region','绘制':'Draw','移动':'Move','缩放':'Resize','导入失败：':'Import failed: ', '布局数据错误：':'Invalid layout: ',
 '归属编辑器 · 实例标签':'Ownership Editor · Instance Labels', '身份种子 · 前景与背景标注':'Identity Seeds · Foreground and Background',
 '角色颜色与名称对照':'Identity Color Legend','显示未绑定 / 已忽略人物':'Show Unbound / Disabled Identities','收起未绑定 / 已忽略人物':'Hide Unbound / Disabled Identities',
 '画笔所属角色':'Brush Identity','画笔半径':'Brush Radius','身份颜色透明度':'Overlay Opacity','角色名称':'Identity Label','加宽节点':'Widen Node','橡皮擦':'Eraser','背景种子':'Background Seed','撤销':'Undo','重做':'Redo','清空笔画':'Clear Strokes','复制笔画':'Copy Strokes','粘贴笔画':'Paste Strokes',
 '身份 · 区域MODEL LoRA组':'Identity · Regional MODEL LoRA Stack','区域MODEL LoRA组':'Regional MODEL LoRA Stack','＋ 添加LoRA':'+ Add LoRA','全部开关':'Toggle All','空组＝仅底模文字；每条只在该身份归属区域作用，不加载全局CLIP。':'Empty stack uses base-model conditioning. Adapters affect their ownership region; no global CLIP adapters are loaded.',
 '选择本地LoRA…':'Select Local LoRA…','LoRA文件':'LoRA File','LoRA用途':'LoRA Purpose','LoRA强度':'LoRA Scale','启用LoRA':'Enable LoRA','删除LoRA':'Remove LoRA','移除此LoRA':'Remove Adapter','质量':'Quality','画风':'Style','服装':'Clothing','其他':'Other','身份':'Identity','强度':'Scale',
 '已忽略 · 底模':'Disabled · Base Model','未绑定 · 底模':'Unbound · Base Model','重复绑定，请检查':'Duplicate Binding',
 '先运行一次区域流程以载入参考图与自动人物轮廓；涂改后重新运行。':'Run the layout source to load the reference and instance regions. Re-run after editing.',
 '在可见人物内部标注前景种子，使用背景种子排除背景。橡皮仅删除标注。笔画仅适用于相同参考图、尺寸与身份数量。':'Mark foreground seeds inside visible identity regions and background seeds outside subjects. Eraser removes annotations. Strokes are valid only for the same reference image, dimensions and identity count.',
 '参考区域变化，已自动清空旧笔画。':'Reference changed; incompatible strokes were cleared. ', '旧笔画已清空。重新运行可载入变化后的结构参考。':'Strokes cleared. Re-run to refresh the reference.',
 '采样完成后显示本阶段运行诊断。':'Runtime diagnostics are displayed after execution.',
 '笔画已复制；可粘贴至同一参考图的种子节点。':'Strokes copied for the same source reference.','浏览器未允许剪贴板；可复制整个种子节点到目标工作流。':'Clipboard access is unavailable. Copy the annotation node instead.',
 '参考图、宽高或人数不同，请先核对原图和设置':'Source image, dimensions or identity count do not match','无效笔画':'Invalid strokes','无法粘贴：':'Cannot paste: ',
 '已保存 ':'Saved ', ' 条笔画；':' strokes; ', '重新运行分割。':'re-run segmentation.', '重新运行后用于采样。':'re-run before sampling.',
 '；已载入全部 ':'; loaded ', ' 人，颜色与角色见上方图例。预览缓存仅供编辑，涂改后重新运行。':' identities. Refer to the color legend. Cached previews are for editing; re-run after changes.',
 '；参考图已载入。标注身份与背景种子后重新运行分割。':'; reference loaded. Annotate identity and background seeds, then re-run segmentation.'
};
export function language(){return String(app.ui?.settings?.getSettingValue?.('Comfy.Locale')??app.extensionManager?.setting?.get?.('Comfy.Locale')??'en').toLowerCase().startsWith('zh')?'zh':'en';}
export function translate(text){if(language()==='zh')return text;let value=text;for(const [cn,en] of Object.entries(PAIRS).sort((a,b)=>b[0].length-a[0].length))value=value.replaceAll(cn,en);return value;}
const subscribers=new Set();let languageTimer;
export function localizeDOM(root,onLanguageChange=()=>{}){
 const values=new WeakMap(),attributes=new WeakMap();let busy=false,closed=false,lastLanguage=language();
 function apply(){if(busy||closed)return;busy=true;const scan=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);let node;
  while(node=scan.nextNode()){
   const parent=node.parentElement;if(['STYLE','SCRIPT'].includes(parent?.tagName))continue;
   if(parent?.tagName==='OPTION'&&parent.value&&parent.parentElement?.dataset.preserveValues&&!['erase','background'].includes(parent.value))continue;
   let st=values.get(node);if(!st||node.nodeValue!==st.last)st={source:node.nodeValue};const text=translate(st.source);st.last=text;values.set(node,st);if(node.nodeValue!==text)node.nodeValue=text;
  }
  for(const element of root.querySelectorAll('[title],[placeholder],[aria-label]')){
   let saved=attributes.get(element);if(!saved){saved={};attributes.set(element,saved);}
   for(const attr of ['title','placeholder','aria-label']){const value=element.getAttribute(attr);if(!value)continue;let st=saved[attr];if(!st||value!==st.last)st={source:value};const next=translate(st.source);st.last=next;saved[attr]=st;if(value!==next)element.setAttribute(attr,next);}
  }
  const changed=lastLanguage!==language();busy=false;lastLanguage=language();if(changed)onLanguageChange();
 }
 const observer=new MutationObserver(()=>queueMicrotask(apply));observer.observe(root,{subtree:true,childList:true,characterData:true});apply();
 const refresh=()=>{if(language()!==lastLanguage)apply();};subscribers.add(refresh);
 if(!languageTimer)languageTimer=setInterval(()=>{for(const fn of subscribers)fn();},500);
 return ()=>{closed=true;observer.disconnect();subscribers.delete(refresh);if(!subscribers.size){clearInterval(languageTimer);languageTimer=null;}};
}
