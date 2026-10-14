"""Version-local SDPA symbolic avoiding Torch 2.7 legacy BF16 scalar-cast bug."""
import torch
from torch.onnx import symbolic_helper as sh
from torch.onnx import _type_utils


@sh.parse_args('v','v','v','v','f','b','v','b')
def sdpa(g,q,k,v,mask=None,dropout=0.,causal=False,scale=None,gqa=False):
    assert not causal and not gqa and dropout==0., 'Only the observed SigLIP inference SDPA contract is supported'
    dtype=_type_utils.JitScalarType.from_value(q).onnx_type()
    q32=g.op('Cast',q,to_i=1);k32=g.op('Cast',k,to_i=1);v32=g.op('Cast',v,to_i=1)
    rank=sh._get_tensor_rank(k)
    perm=list(range(rank));perm[-1],perm[-2]=perm[-2],perm[-1]
    scores=g.op('MatMul',q32,g.op('Transpose',k32,perm_i=perm))
    if sh._is_none(scale):
        shape=g.op('Shape',q)
        dim=g.op('Gather',shape,g.op('Constant',value_t=torch.tensor([-1],dtype=torch.int64)),axis_i=0)
        factor=g.op('Reciprocal',g.op('Sqrt',g.op('Cast',dim,to_i=1)))
    else:
        factor=g.op('Cast',scale,to_i=1)
    scores=g.op('Mul',scores,factor)
    if not sh._is_none(mask):
        if _type_utils.JitScalarType.from_value(mask)==_type_utils.JitScalarType.BOOL:
            mask=g.op('Where',mask,g.op('Constant',value_t=torch.tensor(0.)),
                      g.op('Constant',value_t=torch.tensor(float('-inf'))))
        else:mask=g.op('Cast',mask,to_i=1)
        scores=g.op('Add',scores,mask)
    probs=g.op('Softmax',scores,axis_i=-1)
    out=g.op('MatMul',probs,v32)
    return g.op('Cast',out,to_i=int(dtype))


def register():
    torch.onnx.register_custom_op_symbolic('aten::scaled_dot_product_attention',sdpa,20)


@sh.parse_args('v','v')
def matmul_with_explicit_accumulation(g,a,b):
    dtype=_type_utils.JitScalarType.from_value(a).onnx_type()
    if int(dtype)==16:
        a=g.op('Cast',a,to_i=1);b=g.op('Cast',b,to_i=1)
        return g.op('Cast',g.op('MatMul',a,b),to_i=16)
    return g.op('MatMul',a,b)


def register_attention_rounding():
    # Applies only to aten::matmul (attention QK/probability V), never to Linear
    # weight GEMMs. The original BF16 result boundary is explicitly represented.
    register()
    torch.onnx.register_custom_op_symbolic('aten::matmul',matmul_with_explicit_accumulation,20)
