import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode


class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth):
        super(AggPredictor, self).__init__()

        self.agg_lstm = nn.LSTM(input_size=N_word*2, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.agg_col_name_enc = nn.LSTM(input_size=N_word,
                hidden_size=N_h/2, num_layers=N_depth,
                batch_first=True, dropout=0.3, bidirectional=True)
        self.agg_att = nn.Linear(N_h, N_h)
	self.agg_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(), nn.Linear(N_h, 6))
        self.softmax = nn.Softmax(dim=-1)
        self.agg_out_K = nn.Linear(N_h, N_h)
        self.col_out_col = nn.Linear(N_h, N_h)

    def forward(self, x_emb_var, x_len, col_inp_var=None, col_name_len=None,
                col_len=None, col_num=None, x_type_emb_var=None, gt_sel=None, gt_sel_num=None):
        B = len(x_emb_var)
        max_x_len = max(x_len)
        x_emb_concat = torch.cat((x_emb_var, x_type_emb_var), 2)

        e_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.agg_col_name_enc)
        h_enc, _ = run_lstm(self.agg_lstm, x_emb_concat, x_len)

        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_col[b, x] for x in gt_sel[b]] + [e_col[b, 0]] * (4 - len(gt_sel[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        att_val = torch.matmul(self.agg_att(h_enc).unsqueeze(1), col_emb.unsqueeze(3)).squeeze()  # .transpose(1,2))

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100
        att = self.softmax(att_val.view(B * 4, -1)).view(B, 4, -1)

        K_agg = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        agg_score = self.agg_out(self.agg_out_K(K_agg) + self.col_out_col(col_emb)).squeeze()
        return agg_score
    # def forward(self, x_emb_var, x_len, agg_emb_var, col_inp_var=None,
    #         col_len=None, x_type_emb_var):
    #     B = len(x_emb_var)
    #     max_x_len = max(x_len)
    #
    #     h_enc, _ = run_lstm(self.agg_lstm, x_emb_var, x_len)
    #
    #     agg_enc = self.agg_out_agg(agg_emb_var)
    #     #agg_enc: (B, 6, hid_dim)
    #     #self.sel_att(h_enc) -> (B, max_x_len, hid_dim) .transpose(1, 2) -> (B, hid_dim, max_x_len)
    #     #att_val_agg: (B, 6, max_x_len)
    #     att_val_agg = torch.bmm(agg_enc, self.sel_att(h_enc).transpose(1, 2))
    #
    #     for idx, num in enumerate(x_len):
    #         if num < max_x_len:
    #             att_val_agg[idx, :, num:] = -100
    #
    #     #att_agg: (B, 6, max_x_len)
    #     att_agg = self.softmax(att_val_agg.view((-1, max_x_len))).view(B, -1, max_x_len)
    #     #h_enc.unsqueeze(1) -> (B, 1, max_x_len, hid_dim)
    #     #att_agg.unsqueeze(3) -> (B, 6, max_x_len, 1)
    #     #K_agg_expand -> (B, 6, hid_dim)
    #     K_agg_expand = (h_enc.unsqueeze(1) * att_agg.unsqueeze(3)).sum(2)
    #     #agg_score = self.agg_out(K_agg)
    #     agg_score = self.agg_out_f(self.agg_out_se(agg_emb_var) + self.agg_out_K(K_agg_expand)).squeeze()
    #
    #     return agg_score
