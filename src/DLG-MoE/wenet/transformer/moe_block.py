from typing import Optional, Tuple, Dict
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
import torch
from torch import nn
from wenet.utils.common import get_activation
from wenet.utils.plot_router import plot_tensor_scatter

class MoEBlock(torch.nn.Module):
    """
    Mixture of expert with Positionwise feed forward layer
    See also figure 1 in https://arxiv.org/pdf/2305.15663.pdf
    The output dim is same with the input dim.

    Modified from https://github.com/Lightning-AI/lit-gpt/pull/823
                  https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
        n_expert: number of expert.
        use_gate: Whether or not to use UnSup-Router
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        n_expert: int = 4,
        use_gate: bool = True
    ):
        super(MoEBlock, self).__init__()
        if use_gate:
            self.gate = torch.nn.Linear(idim, n_expert, bias=False)
        else:
            self.gate = torch.nn.Identity()
        self.use_gate = use_gate
        self.experts = torch.nn.ModuleList(
            PositionwiseFeedForward(
                idim, hidden_units, dropout_rate, activation)
            for _ in range(n_expert))
        self.n_expert = n_expert

    def forward(self, xs: torch.Tensor, top_k: int) -> torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        # Includes UnSup-Router, which selects top_k experts from the group and fuses them according to their weights.
        if self.use_gate:
            n_expert_activated = top_k
            router = self.gate(xs)  # (B*L, n_expert)
            logits, selected_experts = torch.topk(router, n_expert_activated)
            weights = torch.nn.functional.softmax(
                logits, dim=1,
                dtype=torch.float).to(dtype=xs.dtype)  # (B*L, n_expert_activated)
            output = torch.zeros_like(xs)  # (B*L, D)
            for i, expert in enumerate(self.experts):
                mask = selected_experts == i
                token_ids, ith_expert = torch.where(mask)
                output[token_ids] += weights[token_ids, ith_expert, None] * expert(
                    xs[token_ids])
        else:  # No UnSup-Router, same weighting for all experts
            expert_outputs = [expert(xs) for expert in self.experts]
            # Average the knowledge extracted from all experts in the group
            output = torch.mean(torch.stack(expert_outputs), dim=0)
        return output


class GroupMoELayer(nn.Module):
    def __init__(self,
                 idim: int,
                 output_dim: int,
                 moe_conf: Dict,
                 dropout_rate: float,
                 gate: torch.nn.Linear,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 ):
        super(GroupMoELayer, self).__init__()
        self.expert_num = moe_conf['expert_num']
        self.num_groups = len(self.expert_num)
        self.is_dynamic = moe_conf['use_dynamic_topk']
        if gate is not None:
            self.language_gate = gate
        else:
            self.language_gate = torch.nn.Linear(idim, self.num_groups)
        self.top_k = moe_conf['top_k']
        activation = get_activation(moe_conf['activation'])
        self.group_moe = nn.ModuleList([MoEBlock(idim=idim,
                                                 hidden_units=moe_conf['ffn_dim'],
                                                 dropout_rate=dropout_rate,
                                                 activation=activation,
                                                 n_expert=self.expert_num[i],
                                                 use_gate=moe_conf['use_gate']) for i in range(self.num_groups)])

    def forward(self, xs, bottle_neck, language: str = 'all', decode_k: int = 1) -> torch.Tensor:
        B, L, D = xs.size(
            )  # batch size, sequence length, embedding dimension (idim)
        xs = xs.view(-1, D)  # (B*L, D)
        router = self.language_gate(bottle_neck).softmax(-1)  # (B*L, n_expert)
        router = router.view(B * L, -1)
        if router.shape[-1] > 3:
            router = router[:, 1:3]  # Discard the blank dimension
        else:
            router = router
        if self.training:
            if self.is_dynamic:  # dynamic top_k training, allows MoE to adapt different top-k values.
                top_k = torch.randint(1, self.top_k + 1, (1,)).item()
            else:
                top_k = self.top_k
            # selected_groups means which language group each frame should go to
            _, selected_groups = torch.topk(router, 1)
        else:  # inference
            top_k = self.top_k
            if language == 'zh':  # specify language, each frame will be sent to the specified language group
                selected_groups = torch.full((xs.shape[0], 1), 0)
            elif language == 'en':
                selected_groups = torch.full((xs.shape[0], 1), 1)
            else:
                _, selected_groups = torch.topk(router, 1)
        # plot_tensor_scatter(selected_groups)
        output = torch.zeros_like(xs)  # (B*L, D)
        for i, moe_block in enumerate(self.group_moe):
            mask = selected_groups == i
            token_ids, _ = torch.where(mask)
            if token_ids.size(0) > 0:  # Ensure there are tokens selected for the expert
                output[token_ids] += moe_block(xs[token_ids], top_k)
        return output.view(B, L, D)


class AdapterLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 ):
        super(AdapterLayer, self).__init__()
        self.gate = nn.Linear(2 * input_dim, 2)
        self.zh_adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.en_adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, xs):
        xs_zh = self.zh_adapter(xs)
        xs_en = self.en_adapter(xs)
        concat = torch.cat((xs_zh, xs_en),dim=-1)
        weights = torch.softmax(self.gate(concat),dim=-1).unsqueeze(-1)
        # 对专家的每一帧输出进行加权组合(加权和连接如何选择？)
        weighted_outputs = [weights[:,:,i] * expert_output for i, expert_output in enumerate(expert_outputs)]
        # 对张量列表在维度0上进行累加
        combined_output = torch.sum(torch.stack(weighted_expert_outputs), dim=0)
        return combined_output, expert_outputs