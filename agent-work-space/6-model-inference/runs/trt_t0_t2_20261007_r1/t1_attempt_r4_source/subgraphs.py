"""Pure tensor interfaces with one vision weight copy and immutable prefix K/V."""
import math
import torch
from torch import nn
from transformers.models.gemma import modeling_gemma as gm


def decoder(model, hidden, mask, positions, prefix=(), cond=None):
    hidden = hidden.to(model.layers[0].self_attn.q_proj.weight.dtype)
    cos, sin = model.rotary_emb(hidden, positions)
    cache = []
    for i, layer in enumerate(model.layers):
        residual = hidden
        x, gate = layer.input_layernorm(hidden, cond)
        attn = layer.self_attn
        shape = (*x.shape[:-1], -1, attn.head_dim)
        q = attn.q_proj(x).view(shape).transpose(1, 2)
        k = attn.k_proj(x).view(shape).transpose(1, 2)
        v = attn.v_proj(x).view(shape).transpose(1, 2)
        q, k = gm.apply_rotary_pos_emb(q, k, cos, sin)
        if prefix:
            k = torch.cat([prefix[2*i], k], dim=2)
            v = torch.cat([prefix[2*i+1], v], dim=2)
        else:
            cache.extend([k, v])
        y, _ = gm.eager_attention_forward(attn, q, k, v, mask, attn.scaling)
        y = attn.o_proj(y.reshape(*x.shape[:-1], -1))
        hidden = gm._gated_residual(residual, y, gate)
        residual = hidden
        x, gate = layer.post_attention_layernorm(hidden, cond)
        hidden = gm._gated_residual(residual, layer.mlp(x), gate)
    hidden, _ = model.norm(hidden, cond)
    return hidden, tuple(cache)


class Vision(nn.Module):
    def __init__(self, net):
        super().__init__()
        p = net.paligemma_with_expert.paligemma.model
        self.vision_tower = p.vision_tower
        self.projector = p.multi_modal_projector

    def forward(self, image):
        # This is exactly PaliGemma.get_image_features, including its projection.
        x = self.vision_tower(image).last_hidden_state
        return self.projector(x)


class Prefix(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.language_model = net.paligemma_with_expert.paligemma.language_model

    def forward(self, image_embeddings, tokens, pad_mask, position_ids, attention_mask):
        text = self.language_model.embed_tokens(tokens) * math.sqrt(2048)
        x = torch.cat([image_embeddings, text], dim=1)
        _, kv = decoder(self.language_model, x, attention_mask, position_ids)
        return (pad_mask, *kv)


class ActionStep(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.expert = net.paligemma_with_expert.gemma_expert.model
        self.action_in = net.action_in_proj
        self.action_out = net.action_out_proj
        self.time_in = net.time_mlp_in
        self.time_out = net.time_mlp_out

    def forward(self, noisy_actions, time_embedding, position_ids, attention_mask, *prefix):
        cond = torch.nn.functional.silu(self.time_out(torch.nn.functional.silu(self.time_in(time_embedding))))
        hidden, _ = decoder(self.expert, self.action_in(noisy_actions), attention_mask, position_ids, prefix, cond)
        return self.action_out(hidden[:, -10:].float())


def inputs(net, observation):
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    images, image_masks, tokens, token_masks, state = net._preprocess_observation(observation, train=False)
    embeddings = [net.paligemma_with_expert.embed_image(im) for im in images]
    pad = torch.cat([*(m[:, None].expand(1, e.shape[1]) for m, e in zip(image_masks, embeddings)), token_masks], 1)
    pos = torch.cumsum(pad, 1) - 1
    mask = net._prepare_attention_masks_4d(make_att_2d_masks(pad, torch.zeros_like(pad)))
    suffix_mask = torch.cat([pad[:, None, :].expand(1, 10, pad.shape[1]),
                              torch.ones(1, 10, 10, dtype=torch.bool, device=pad.device)], 2)
    action_pos = pad.sum(-1)[:, None] + torch.arange(10, device=pad.device)[None, :]
    return images, (torch.cat(embeddings, 1), tokens, pad, pos, mask), (action_pos, net._prepare_attention_masks_4d(suffix_mask))
