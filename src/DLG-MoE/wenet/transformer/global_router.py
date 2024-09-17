import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
from wenet.transformer.embedding import PositionalEncoding
from wenet.utils.common import get_activation


class FrameLID(torch.nn.Module):
    """CTC module"""
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lid = torch.nn.Linear(eprojs, odim)
        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lid(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat_prob = ys_hat.softmax(2)
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        # ys_hat_test=ys_hat.softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        return loss

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lid(hs_pad), dim=2)


class SegmentLid(nn.Module):
    def __init__(self, input_size, num_class):
        super(SegmentLid, self).__init__()
        self.lid_net = torch.nn.Linear(input_size, num_class)
        self.lid_criterion = LIDLabelSmoothingLoss(
            classes=3,
            smoothing=0.1,
            dim=-1,
        )

    def forward(self, x, utt_id):
        global_avg_pool_output = torch.mean(x, dim=1, keepdim=True)
        # global_avg_pool_output = self.asp(x)
        encoder_lid = self.lid_net(F.dropout(global_avg_pool_output, p=0.1))
        encoder_lid = encoder_lid.squeeze(dim=1)
        utt_id = utt_id.squeeze(dim=1)
        prob_lid = F.softmax(encoder_lid, dim=1)
        _, max_indices = torch.max(prob_lid, dim=1)
        correct_predictions = max_indices == utt_id
        num_correct = correct_predictions.sum().item()
        total_samples = max_indices.size(0)
        lid_acc = num_correct / total_samples
        acc_lid = torch.tensor(lid_acc)
        log_prob_lid = torch.log(prob_lid)
        lid_loss = self.lid_criterion(log_prob_lid, utt_id)
        lid_loss = 10 * lid_loss
        return lid_loss, acc_lid

    def get_gate(self, x):
        with torch.no_grad():
            global_avg_pool_output = torch.mean(x, dim=1, keepdim=True)
            encoder_lid = self.lid_net(F.dropout(global_avg_pool_output, p=0.1))
            encoder_lid = encoder_lid.squeeze(dim=1)
            prob_lid = F.softmax(encoder_lid, dim=1) 
            _, top_k_indices = torch.topk(prob_lid, 1, dim=-1) 
        return top_k_indices

    def _forward_lid(self, x): 
        global_avg_pool_output = torch.mean(x, dim=1, keepdim=True)
        encoder_lid = self.lid_net(F.dropout(global_avg_pool_output, p=0.1))
        encoder_lid = encoder_lid.squeeze(dim=1)
        prob_lid = F.softmax(encoder_lid, dim=1)
        max_values, max_indices = torch.max(prob_lid, dim=1)
        return max_indices


class LIDPromptEncoder(torch.nn.Module):
    def __init__(
        self,
        frameLID: FrameLID,
        encoder_output_size: int,
        lid_class: int,
    ):
        super().__init__()
        self.frameLID = frameLID
        self.embedding = torch.nn.Embedding(lid_class, encoder_output_size)
        self.layer_norm = nn.LayerNorm(encoder_output_size, eps=1e-5)

    def forward(self, xs, tab_clone):
        lid_embedding = self.embedding(tab_clone)
        xs = xs + lid_embedding
        xs = self.layer_norm(xs)
        return xs

class LIDPromptConcat(torch.nn.Module):
    def __init__(
        self,
        frameLID: FrameLID,
        encoder_output_size: int,
        lid_class: int,
    ):
        super().__init__()
        self.frameLID = frameLID
        self.concat_linear = torch.nn.Linear(lid_class + encoder_output_size, encoder_output_size)
        self.activation = get_activation("swish")
        self.frame_lid = frameLID
        self.layer_norm = nn.LayerNorm(encoder_output_size, eps=1e-5)

    def forward(self, xs):
        logits = self.frame_lid.ctc_lid(xs)
        logits = self.activation(logits)
        xs_concat = torch.cat((xs, logits), dim=-1)
        xs = self.concat_linear(xs_concat)
        xs = self.layer_norm(xs)
        return xs


class LIDPromptAdapter(torch.nn.Module):
    def __init__(
        self,
        frameLID: FrameLID,
        encoder_output_size: int,
        lid_class: int,
    ):
        super().__init__()
        self.frameLID = frameLID
        self.up_linear = torch.nn.Linear(lid_class, encoder_output_size)
        self.activation = get_activation("relu")
        self.frame_lid = frameLID
        self.layer_norm = nn.LayerNorm(encoder_output_size, eps=1e-5)

    def forward(self, xs):
        residual = xs
        logits = self.frame_lid.ctc_lid(xs)
        logits = self.activation(logits)
        up_logits = self.up_linear(logits)
        up_logits = self.layer_norm(up_logits)
        xs = residual + up_logits
        return xs
