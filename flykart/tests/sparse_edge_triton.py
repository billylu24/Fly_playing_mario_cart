"""Fused gradient for existing edge values; no dense N×N intermediate."""
import torch
import triton
import triton.language as tl


@triton.jit
def _edge_grad(G, H, ROW, COL, OUT, E:tl.constexpr, B:tl.constexpr,
               GS0:tl.constexpr, GS1:tl.constexpr, HS0:tl.constexpr, HS1:tl.constexpr,
               BLOCK:tl.constexpr, BB:tl.constexpr):
    edges=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    batches=tl.arange(0,BB)
    row=tl.load(ROW+edges,edges<E,other=0)
    col=tl.load(COL+edges,edges<E,other=0)
    mask=(edges[:,None]<E)&(batches[None,:]<B)
    g=tl.load(G+row[:,None]*GS0+batches[None,:]*GS1,mask,other=0)
    h=tl.load(H+col[:,None]*HS0+batches[None,:]*HS1,mask,other=0)
    value=tl.sum(g*h,axis=1)
    tl.store(OUT+edges,value,edges<E)


def edge_grad(grad_out,h,rows,cols):
    out=torch.empty(rows.numel(),device=h.device,dtype=h.dtype)
    _edge_grad[(triton.cdiv(rows.numel(),256),)](
        grad_out,h,rows,cols,out,rows.numel(),h.shape[1],
        grad_out.stride(0),grad_out.stride(1),h.stride(0),h.stride(1),
        BLOCK=256,BB=triton.next_power_of_2(h.shape[1]))
    return out
