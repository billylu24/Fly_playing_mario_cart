"""First-order sparse matmul prototype; explicit gradients only on existing edges."""
import torch


class Graph:
    """Fixed square topology; indices must be sorted, unique, valid COO indices.

    Callers coalesce input once before construction. This research prototype
    supports first derivatives only, fixed topology, and floating dense inputs.
    """
    def __init__(self, indices, n):
        self.n = n
        self.rows, self.cols = indices[0].contiguous(), indices[1].contiguous()
        self.crow = torch.cat([torch.zeros(1, dtype=torch.int64, device=indices.device),
                               torch.bincount(self.rows, minlength=n).cumsum(0)])
        self.perm = torch.argsort(self.cols * n + self.rows)
        self.tcols = self.rows[self.perm]
        self.tcrow = torch.cat([torch.zeros(1, dtype=torch.int64, device=indices.device),
                                torch.bincount(self.cols, minlength=n).cumsum(0)])
        self.indices = indices

    def csr(self, values):
        return torch.sparse_csr_tensor(self.crow, self.cols, values,
                                       size=(self.n, self.n), check_invariants=False)


class EdgeMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, h, graph, backend):
        ctx.graph = graph
        ctx.backend = backend
        ctx.save_for_backward(values, h)
        return torch.sparse.mm(graph.csr(values), h)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        values, h = ctx.saved_tensors
        graph = ctx.graph
        grad_out = grad_out.contiguous()
        grad_values = grad_h = None
        if ctx.needs_input_grad[0]:
            if ctx.backend == "triton":
                from sparse_edge_triton import edge_grad
                grad_values = edge_grad(grad_out,h,graph.rows,graph.cols)
            else:
                grad_values = torch.empty_like(values)
                # Bound temporary gather tensors to chunk_size × batch, not N × N.
                for start in range(0, values.numel(), 262144):
                    end = min(start + 262144, values.numel())
                    grad_values[start:end] = (
                        grad_out.index_select(0, graph.rows[start:end]) *
                        h.index_select(0, graph.cols[start:end])).sum(dim=1)
        if ctx.needs_input_grad[1]:
            wt = torch.sparse_csr_tensor(graph.tcrow, graph.tcols, values[graph.perm],
                                         size=(graph.n, graph.n), check_invariants=False)
            grad_h = torch.sparse.mm(wt, grad_out)
        return grad_values, grad_h, None, None


def multiply(kind, values, h, graph):
    if kind in ["edge", "edge_triton"]:
        return EdgeMatmul.apply(values, h, graph, "triton" if kind=="edge_triton" else "torch")
    if kind == "csr":
        return torch.sparse.mm(graph.csr(values), h)
    if kind == "coo":
        w = torch.sparse_coo_tensor(graph.indices, values, (graph.n, graph.n), is_coalesced=True)
        return torch.sparse.mm(w, h)
    raise ValueError(kind)
