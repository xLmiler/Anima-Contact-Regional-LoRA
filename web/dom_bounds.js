/* 使用节点画布坐标约束 DOM，避免选择节点时宿主沿用过期的 widget.width。 */
export function containWidget(node,widget,{height,minHeight=height,maxHeight=height}={}){
 const element=widget.element;
 Object.assign(element.style,{width:'100%',maxWidth:'100%',minWidth:'0',boxSizing:'border-box',contain:'inline-size',overflowX:'auto'});
 widget.options||={};
 const value=x=>typeof x==='function'?x():x;
 widget.options.getMinHeight=()=>value(minHeight);
 widget.options.getMaxHeight=()=>value(maxHeight);
 const currentWidth=()=>Math.max(0,Number(node.size?.[0])||0);
 // DomWidgets.vue 在 selectionChanged 时优先读取 width；不可保留旧的固定值。
 Object.defineProperty(widget,'width',{configurable:true,get:currentWidth,set(){}});
 widget.computeSize=()=>[currentWidth(),value(height)];
 widget.computeLayoutSize=()=>({minWidth:0,minHeight:value(minHeight),maxHeight:value(maxHeight)});
 return widget;
}
