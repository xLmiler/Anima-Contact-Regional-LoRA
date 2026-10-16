"""Chunked spatial self-attention isolation on native Anima Q/K/V tensors."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def isolated_attention(q,k,v,gates,strength,frames=1,chunk_size=256):
 """Keep background connectivity; suppress only different owned identities."""
 q,k,v=[x if torch.is_tensor(x) else x.take() if hasattr(x,'take') else x for x in (q,k,v)]
 if not all(torch.is_tensor(x) and x.ndim>=4 for x in (q,k,v)):raise RuntimeError('Unsupported native Anima attention tensor layout')
 batch=q.shape[0];heads=q.shape[-2];dim=q.shape[-1]
 q=q.reshape(batch,-1,heads,dim);k=k.reshape(batch,-1,heads,dim);v=v.reshape(batch,-1,heads,dim)
 if q.shape[1]!=k.shape[1] or q.shape[1]!=int(frames)*gates[0].numel():raise RuntimeError('Self-attention token count does not match the ownership grid')
 masks=gates.reshape(len(gates),-1).repeat(1,int(frames));peak,owner=masks.max(0);owner=torch.where(peak>.05,owner,torch.full_like(owner,-1)).to(q.device)
 qt=q.transpose(1,2);kt=k.transpose(1,2);vt=v.transpose(1,2);pieces=[]
 penalty=float('-inf') if strength>=1 else float(torch.log(torch.tensor(max(1e-6,1-float(strength)))))
 for start in range(0,q.shape[1],chunk_size):
  ids=owner[start:start+chunk_size];blocked=(ids[:,None]!=owner[None,:])&(ids[:,None]>=0)&(owner[None,:]>=0)
  bias=torch.zeros(blocked.shape,device=q.device,dtype=q.dtype).masked_fill(blocked,penalty)
  pieces.append(F.scaled_dot_product_attention(qt[:,:,start:start+chunk_size],kt,vt,attn_mask=bias[None,None]))
 out=torch.cat(pieces,dim=2).transpose(1,2)
 return out.reshape(batch,q.shape[1],heads*dim)
