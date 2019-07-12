import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode

class SelCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, db_content):
        super(SelCondPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        if db_content == 0:
            in_size = N_word+N_word/2
        else:
            in_size = N_word+N_word

        self.selcond_lstm = nn.LSTM(input_size=in_size, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.selcond_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h / 2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.ty_sel_num_out = nn.Linear(N_h, N_h)
        self.sel_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 4))
        self.sel_num_type_att = nn.Linear(N_h, N_h)

        self.sel_att = nn.Linear(N_h, N_h)
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.ty_cond_num_out = nn.Linear(N_h, N_h)
        self.cond_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 5))
        self.cond_num_type_att = nn.Linear(N_h, N_h)

        self.cond_col_att = nn.Linear(N_h, N_h)
        self.cond_col_out_K = nn.Linear(N_h, N_h)
        self.cond_col_out_col = nn.Linear(N_h, N_h)
        self.cond_col_out_sel = nn.Linear(N_h, N_h)
        self.col_att = nn.Linear(N_h, N_h)
        self.cond_col_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax() #dim=1


    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len, x_type_emb_var, gt_sel):
        max_x_len = max(x_len)
        max_col_len = max(col_len)
        B = len(x_len)

        x_emb_concat = torch.cat((x_emb_var, x_type_emb_var), 2)
        e_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.selcond_name_enc)
        # e_col, _ = run_lstm(self.selcond_name_enc, col_inp_var, col_len)
        h_enc, _ = run_lstm(self.selcond_lstm, x_emb_concat, x_len)

        # Predict the number of selected columns
        # att_sel_num_type_val:(B, max_col_len, max_x_len)
        att_sel_num_type_val = torch.bmm(e_col, self.sel_num_type_att(h_enc).transpose(1, 2))

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                att_sel_num_type_val[idx, num:, :] = -100
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_sel_num_type_val[idx, :, num:] = -100

        # att_sel_num_type: (B, max_col_len, max_x_len)
        att_sel_num_type = self.softmax(att_sel_num_type_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        # h_enc.unsqueeze(1): (B, 1, max_x_len, hid_dim)
        # att_sel_num_type.unsqueeze(3): (B, max_col_len, max_x_len, 1)
        # K_num_type (B, max_col_len, hid_dim)
        K_sel_num_type = (h_enc.unsqueeze(1) * att_sel_num_type.unsqueeze(3)).sum(2).sum(1)
        # K_sel_num: (B, hid_dim)
        # K_sel_num_type (B, hid_dim)
        sel_num_score = self.sel_num_out(self.ty_sel_num_out(K_sel_num_type))

        #Predict the selection condition
        #att_val: (B, max_col_len, max_x_len)
        sel_att_val = torch.bmm(e_col, self.sel_att(h_enc).transpose(1, 2))
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                sel_att_val[idx, :, num:] = -100
        sel_att = self.softmax(sel_att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        #K_sel_expand -> (B, max_number of col names in batch tables, hid_dim)
        K_sel_expand = (h_enc.unsqueeze(1) * sel_att.unsqueeze(3)).sum(2)
        sel_score = self.sel_out(self.sel_out_K(K_sel_expand) + \
                self.sel_out_col(e_col)).squeeze()

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                sel_score[idx, num:] = -100

        # Predict the number of conditions
        #att_cond_num_type_val:(B, max_col_len, max_x_len)
        att_cond_num_type_val = torch.bmm(e_col, self.cond_num_type_att(h_enc).transpose(1, 2))

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                att_cond_num_type_val[idx, num:, :] = -100
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_cond_num_type_val[idx, :, num:] = -100

        #att_cond_num_type: (B, max_col_len, max_x_len)
        att_cond_num_type = self.softmax(att_cond_num_type_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        #h_enc.unsqueeze(1): (B, 1, max_x_len, hid_dim)
        #att_cond_num_type.unsqueeze(3): (B, max_col_len, max_x_len, 1)
        #K_num_type (B, max_col_len, hid_dim)
        K_cond_num_type = (h_enc.unsqueeze(1) * att_cond_num_type.unsqueeze(3)).sum(2).sum(1)
        #K_cond_num: (B, hid_dim)
        #K_cond_num_type (B, hid_dim)
        cond_num_score = self.cond_num_out(self.ty_cond_num_out(K_cond_num_type))

        #Predict the columns of conditions
        if gt_sel is None:
            num = np.argmax(sel_num_score.data.cpu().numpy(), axis=1) + 1
            sel = sel_score.data.cpu().numpy()
            # gt_sel = np.argmax(sel_score.data.cpu().numpy(), axis=1)
            chosen_sel_col_gt = [list(np.argsort(-sel[b])[:num[b]]) for b in range(len(num))]
        else:
            chosen_sel_col_gt = [[x[0] for x in one_gt_sel] for one_gt_sel in gt_sel]

            sel_col_emb = []
            for b in range(B):
                cur_sel_col_emb = torch.stack([e_col[b, x]
                                               for x in chosen_sel_col_gt[b]] + [e_col[b, 0]] *
                                              (4 - len(chosen_sel_col_gt[b])))  # Pad the columns to maximum (4)
                sel_col_emb.append(cur_sel_col_emb)
            sel_col_emb = torch.stack(sel_col_emb)

        # chosen_sel_idx = torch.LongTensor(gt_sel)
        #aux_range (B) (0,1,...)
        # aux_range = torch.LongTensor(range(len(gt_sel)))
        # if x_emb_var.is_cuda:
        #     chosen_sel_idx = chosen_sel_idx.cuda()
        #     aux_range = aux_range.cuda()
        #chosen_e_col: (B, hid_dim)
        # chosen_e_col = e_col[aux_range, chosen_sel_idx]
        #chosen_e_col.unsqueeze(2): (B, hid_dim, 1)
        #self.col_att(h_enc): (B, max_x_len, hid_dim)
        #att_sel_val: (B, max_x_len)

        # K_agg = (h_enc.unsqueeze(1) * agg_att.unsqueeze(3)).sum(2)
        #
        # agg_score = self.agg_out(self.agg_out_K(K_agg) + self.col_out_col(sel_col_emb)).squeeze()
        #
        #
        # att_sel_val = torch.bmm(self.col_att(h_enc), chosen_e_col.unsqueeze(2)).squeeze()
        att_sel_val = torch.matmul(self.col_att(h_enc).unsqueeze(1),
                                   sel_col_emb.unsqueeze(3)).squeeze()
        col_att_val = torch.bmm(e_col, self.cond_col_att(h_enc).transpose(1, 2))
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                col_att_val[idx, :, num:] = -100
                att_sel_val[idx, num:] = -100

        sel_att = self.softmax(att_sel_val.view(B * 4, -1)).view(B, 4, -1)

        #K_sel_agg = (h_enc * sel_att.unsqueeze(2).expand_as(h_enc)).sum(1)
        K_sel_agg = (h_enc.unsqueeze(1) * sel_att.unsqueeze(3)).sum(2)

        col_att = self.softmax(col_att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        K_cond_col = (h_enc.unsqueeze(1) * col_att.unsqueeze(3)).sum(2)

        cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col)
                + self.cond_col_out_col(e_col)
                + self.cond_col_out_sel(K_sel_agg.unsqueeze(1).expand_as(K_cond_col))).squeeze()

        for b, num in enumerate(col_len):
            if num < max_col_len:
                cond_col_score[b, num:] = -100

        sel_cond_score = (sel_num_score, cond_num_score, sel_score, cond_col_score)

        return sel_cond_score
