import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules.word_embedding import WordEmbedding
from modules.aggregator_predict import AggPredictor
from modules.sel_condition_predict import SelCondPredictor
from modules.condtion_op_str_predict import CondOpStrPredictor
from modules.select_number import SelNumPredictor
from modules.where_relation import WhereRelationPredictor
from modules.selection_predict import SelPredictor
from modules.sqlnet_condition_predict import SQLNetCondPredictor


class SQLNet(nn.Module):
    def __init__(self, word_emb, N_word, N_h=120, N_depth=2,
           use_ca=True, gpu=True, trainable_emb=False, db_content=0):
        super(SQLNet, self).__init__()
        self.trainable_emb = trainable_emb
        self.db_content = db_content

	self.use_ca = use_ca
        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'OR',
                '==', '>', '<', '!=', '<BEG>']
        self.COND_OPS = ['>', '<', '==', '!=']

        #the model actually doesn't use type embedding when db_content == 1
        if db_content == 0:
            is_train = True
        else:
            is_train = False

        # self.sel_num_type_embed_layer = WordEmbedding(word_emb, N_word, gpu,
        #         self.SQL_TOK, trainable=is_train)
        self.agg_type_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                self.SQL_TOK, trainable=is_train)
        self.sel_type_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                self.SQL_TOK, trainable=is_train)
        self.cond_type_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                self.SQL_TOK, trainable=is_train)
        self.where_rela_type_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                 self.SQL_TOK, trainable=is_train)

        self.embed_layer = WordEmbedding(word_emb, N_word, gpu,
                self.SQL_TOK, trainable=trainable_emb)

        # # Predict selected column number
        # self.sel_num = SelNumPredictor(N_word, N_h, N_depth)
        #
        # # Predict which columns are selected
        # self.sel_pred = SelPredictor(N_word, N_h, N_depth, self.max_tok_num, use_ca=use_ca)

        #Predict aggregator
        self.agg_pred = AggPredictor(N_word, N_h, N_depth)

        # # Predict number of conditions, condition columns, condition operations and condition values
        # self.cond_pred = SQLNetCondPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num,use_ca, gpu, db_content)

        #Predict selected column number + select column + condition number and columns
        self.selcond_pred = SelCondPredictor(N_word, N_h, N_depth, gpu, db_content)

        #Predict condition operators and string values
        self.op_str_pred = CondOpStrPredictor(N_word, N_h, N_depth,
                self.max_col_num, self.max_tok_num, gpu, db_content)

        # Predict conditions' relation
        self.where_rela_pred = WhereRelationPredictor(N_word, N_h, N_depth, use_ca=use_ca)

        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        if gpu:
            self.cuda()


    def get_str_index(self, all_toks, this_str):
        cur_seq = []
        tok_gt_1 = [t for t in all_toks if len(t) > 1]
        if this_str in all_toks:
            all_str = [['<BEG>'], this_str, ['<END>']]
            cur_seq = [all_toks.index(s) if s in all_toks else 0 for s in all_str]
        elif len(tok_gt_1) > 0:
            flag = False
            for tgt in tok_gt_1:
                if set(tgt).issubset(this_str):
                    not_tgt = [x for x in this_str if x not in tgt]
                    if len(not_tgt) > 0:
                        not_tgt = [[x] for x in not_tgt]
                        all_str = [tgt] + not_tgt
                    else:
                        all_str = [tgt]
                    beg_ind = all_toks.index(['<BEG>']) if ['<BEG>'] in all_toks else 0
                    end_ind = all_toks.index(['<END>']) if ['<END>'] in all_toks else 0
                    cur_seq = sorted([all_toks.index(s) if s in all_toks else 0 for s in all_str])
                    cur_seq = [beg_ind] + cur_seq + [end_ind]
                elif set(this_str).issubset(tgt):
                    all_str = [['<BEG>'], tgt, ['<END>']]
                    cur_seq = [all_toks.index(s) if s in all_toks else 0 for s in all_str]

                if len(cur_seq) > 0:
                    flag = True
                    break

            if not flag:
                all_str = [['<BEG>']] + [[x] for x in this_str] + [['<END>']]
                cur_seq = [all_toks.index(s) if s in all_toks else 0 for s in all_str]
        else:
            all_str = [['<BEG>']] + [[x] for x in this_str] + [['<END>']]
            cur_seq = [all_toks.index(s) if s in all_toks else 0 for s in all_str]

        return cur_seq

    def generate_gt_where_seq_test(self, q, gt_cond_seq):
        ret_seq = []
        for cur_q, ans in zip(q, gt_cond_seq):
	    q_toks = []
	    q_toks_cnt = []
	    cur_q_join = []
            cnt=0
	    for toks in cur_q:
		cur_q_join.append(u"".join(toks))
		cnt1 = 0
		for tok in toks:
		    q_toks.append(tok)
		    cnt1 = cnt1 + len(tok)
		cnt = cnt + cnt1
		q_toks_cnt.append(cnt)
	    #for tok in q_toks:
	    #   print("{}".format(tok.encode('utf-8')))
	    #print("q_toks:{}".format(len(q_toks)))
	    temp_q = u"".join(cur_q_join)
	    #print("temp_q:{}".format(temp_q.encode('utf-8')))
            #cur_q = [u'<BEG>'] + cur_q + [u'<END>']
            #cur_q = [u'<BEG>'] + cur_q_join + [u'<END>']
            #print("cur_q:{}".format(cur_q.encode('utf-8')))
	    record = []
            record_cond = []
            for cond in ans:
                #print("cond[2]:{}".format(cond[2].encode('utf-8')))
		if cond[2] not in temp_q:
                    record.append((False, cond[2]))
                else:
                    record.append((True, cond[2]))
            for idx, item in enumerate(record):
                temp_ret_seq = []
                if item[0]:
                    temp_ret_seq.append(0)
		    start_idx = -1
		    end_idx = -1
                    start_idx_org = temp_q.index(item[1])+1
		    end_idx_org = start_idx_org+len(item[1])-1
		    for idx, cnt in enumerate(q_toks_cnt):
			if start_idx_org <= cnt:
				start_idx=idx
				break
		    for idx, cnt in enumerate(q_toks_cnt):
			if end_idx_org <= cnt:
				end_idx = idx + 1
				break
		    if end_idx == -1:
			end_idx = len(q_toks_cnt)+1
		    #temp_ret_seq.extend(list(range(temp_q.index(item[1])+1,temp_q.index(item[1])+len(item[1])+1)))
		    #print("start_idx:{} end_idx:{}".format(start_idx, end_idx))
		    temp_ret_seq.extend(list(range(start_idx+1, end_idx+1))) 
		    temp_ret_seq.append(len(q_toks_cnt)+1)
                else:
                    temp_ret_seq.extend([0,len(q_toks_cnt)+1])
                #print("temp_ret_sql:{}".format(temp_ret_seq))
		record_cond.append(temp_ret_seq)
            ret_seq.append(record_cond)
        return ret_seq


    def generate_gt_where_seq(self, q, col):
        ret_seq = []
        for cur_q, cur_query in zip(q, col):
            cur_values = []
            st = cur_query.index(u'WHERE') + 1 if \
                u'WHERE' in cur_query else len(cur_query)
            all_toks = [['<BEG>']] + cur_q + [['<END>']]
            while st < len(cur_query):
                ed = len(cur_query) if 'AND' not in cur_query[st:] \
                    else cur_query[st:].index('AND') + st
                if '==' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('==') + st
                elif '>' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('>') + st
                elif '<' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('<') + st
                elif '>=' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('>=') + st
                elif '<=' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('<=') + st
                elif '!=' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('!=') + st
                else:
                    raise RuntimeError("No operator in it!")

                this_str = cur_query[op + 1:ed]
                cur_seq = self.get_str_index(all_toks, this_str)
                cur_values.append(cur_seq)
                st = ed + 1
            ret_seq.append(cur_values)
        # ret_seq = []
        # for cur_q, ans in zip(q, gt_cond_seq):
        #     temp_q = u"".join(cur_q)
        #     cur_q = [u'<BEG>'] + cur_q + [u'<END>']
        #     record = []
        #     record_cond = []
        #     for cond in ans:
        #         if cond[2] not in temp_q:
        #             record.append((False, cond[2]))
        #         else:
        #             record.append((True, cond[2]))
        #     for idx, item in enumerate(record):
        #         temp_ret_seq = []
        #         if item[0]:
        #             temp_ret_seq.append(0)
        #             temp_ret_seq.extend(list(range(temp_q.index(item[1])+1,temp_q.index(item[1])+len(item[1])+1)))
        #             temp_ret_seq.append(len(cur_q)-1)
        #         else:
        #             temp_ret_seq.append([0,len(cur_q)-1])
        #         record_cond.append(temp_ret_seq)
        #     ret_seq.append(record_cond)
        return ret_seq


    def forward(self, q, col, col_num, q_type, col_type,
            gt_where = None, gt_cond=None, gt_sel=None, gt_sel_num=None):
        B = len(q)
        # pred_sel_num, pred_agg, pred_sel, pred_cond, pred_where_rela = pred_entry
        pred_agg=True
        pred_sel = True
        pred_cond = True
        pred_sel_num = True
        pred_where_rela = True

        agg_score = None
        sel_cond_score = None
        cond_op_str_score = None

        if self.trainable_emb:
            if pred_agg:
                x_emb_var, x_len = self.agg_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.agg_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num, gt_sel=gt_sel)

            if pred_sel:
                x_emb_var, x_len = self.sel_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.sel_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                sel_score = self.selcond_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num)

            if pred_cond:
                x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.cond_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num,
                        gt_where, gt_cond)
        elif self.db_content == 0:
            x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col, is_list=True, is_q=True)
            #col_inp_var, col_len = self.embed_layer.gen_x_batch(col, col, is_list=True)
            col_inp_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col)
            agg_emb_var = self.embed_layer.gen_agg_batch(q)
            max_x_len = max(x_len)
            if pred_sel_num and pred_agg and pred_sel:
                sel_num_score = self.sel_num(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)
            if gt_sel_num:
                pre_sel_num = gt_sel_num
            else:
                pr_sel_num = np.argmax(sel_num_score.data.cpu().numpy(), axis=1)
            x_type_sel_emb_var, _ = self.sel_type_embed_layer.gen_xc_type_batch(q_type, is_list=True)
            sel_cond_score = self.selcond_pred(x_emb_var, x_len, col_inp_var, col_len, x_type_sel_emb_var, gt_sel)

            if gt_sel:
                pr_sel = gt_sel
            else:
                num = np.argmax(sel_num_score.data.cpu().numpy(), axis=1)
		sel = sel_cond_score.data.cpu().numpy()
		pr_sel = [list(np.argsort(-sel[b])[:num[b]]) for b in range(len(num))]
            agg_score = self.agg_pred(x_emb_var, x_len, agg_emb_var, col_inp_var, col_len, gt_sel=pr_sel,
                                      gt_sel_num=pr_sel_num)
            # if pred_agg:
            #     #x_type_agg_emb_var, _ = self.agg_type_embed_layer.gen_xc_type_batch(q_type, is_list=True)
            #     agg_score = self.agg_pred(x_emb_var, x_len, agg_emb_var, col_inp_var, col_len)
            #
            # if pred_sel:
            #     x_type_sel_emb_var, _ = self.sel_type_embed_layer.gen_xc_type_batch(q_type, is_list=True)
            #     sel_cond_score = self.selcond_pred(x_emb_var, x_len, col_inp_var, col_len, x_type_sel_emb_var,
            #                                    gt_sel)

            if pred_cond:
                x_type_cond_emb_var, _ = self.cond_type_embed_layer.gen_xc_type_batch(q_type, is_list=True)
                cond_op_str_score = self.op_str_pred(x_emb_var, x_len, col_inp_var, col_len, x_type_cond_emb_var,
                                                     gt_where, gt_cond, sel_cond_score)

            if pred_where_rela:
                where_rela_score = self.where_rela_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)
        else:
            x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col, is_list=True, is_q=True)
            col_inp_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col)
            x_type_emb_var, x_type_len = self.embed_layer.gen_x_batch(q_type, col, is_list=True, is_q=True)
            #col_type_inp_var, col_type_len = self.embed_layer.gen_x_batch(col_type, col_type, is_list=True)

            sel_cond_score = self.selcond_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, x_type_emb_var, gt_sel)
            agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var, col_len, col_name_len, x_type_emb_var, gt_sel, sel_cond_score)
            cond_op_str_score = self.op_str_pred(x_emb_var, x_len, col_inp_var, col_len, col_name_len, x_type_emb_var, gt_where, gt_cond, sel_cond_score)
            where_rela_score = self.where_rela_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num,
                                                    x_type_emb_var)

            # sel_num_score = self.sel_num(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, x_type_emb_var)
            #
            # if gt_sel_num:
            #     pr_sel_num = gt_sel_num
            # else:
            #     pr_sel_num = np.argmax(sel_num_score.data.cpu().numpy(), axis=1)
	        # sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, x_type_emb_var)
            #
            # if gt_sel:
            #     pr_sel = gt_sel
            # else:
            #     num = np.argmax(sel_num_score.data.cpu().numpy(), axis=1) +1
		    # sel = sel_score.data.cpu().numpy()
		    # pr_sel = [list(np.argsort(-sel[b])[:num[b]]) for b in range(len(num))]
            # agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, x_type_emb_var, gt_sel=pr_sel,
            #                       gt_sel_num=pr_sel_num)
            # cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, x_type_emb_var, gt_where, gt_cond)
            # where_rela_score = self.where_rela_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, x_type_emb_var)
        return (sel_cond_score, agg_score, cond_op_str_score, where_rela_score)

    def loss(self, score, truth_num, gt_where):
        sel_cond_score, agg_score, cond_op_str_score, where_rela_score = score

        sel_num_score, cond_num_score, sel_score, cond_col_score = sel_cond_score
        cond_op_score, cond_str_score = cond_op_str_score

        B = len(truth_num)
        loss = 0

        # Evaluate select number
        sel_num_truth = map(lambda x: x[0], truth_num)
        sel_num_truth = torch.from_numpy(np.array(sel_num_truth))
        if self.gpu:
            sel_num_truth = Variable(sel_num_truth.cuda())
        else:
            sel_num_truth = Variable(sel_num_truth)
        loss += self.CE(sel_num_score, sel_num_truth)

        # Evaluate select column
        T = len(sel_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][list(truth_num[b][1])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            sel_col_truth_var = Variable(data.cuda())
        else:
            sel_col_truth_var = Variable(data)
        sigm = nn.Sigmoid()
        sel_col_prob = sigm(sel_score)
        bce_loss = -torch.mean(
            3 * (sel_col_truth_var * torch.log(sel_col_prob + 1e-10)) +
            (1 - sel_col_truth_var) * torch.log(1 - sel_col_prob + 1e-10)
        )
        loss += bce_loss

        # Evaluate select aggregation
        for b in range(len(truth_num)):
            data = torch.from_numpy(np.array(truth_num[b][2]))
            if self.gpu:
                sel_agg_truth_var = Variable(data.cuda())
            else:
                sel_agg_truth_var = Variable(data)
            sel_agg_pred = agg_score[b, :len(truth_num[b][1])]
            loss += (self.CE(sel_agg_pred, sel_agg_truth_var)) / len(truth_num)

        # Evaluate the number of conditions
        cond_num_truth = map(lambda x: x[3], truth_num)
        data = torch.from_numpy(np.array(cond_num_truth))
        if self.gpu:
            try:
                cond_num_truth_var = Variable(data.cuda())
            except:
                print "cond_num_truth_var error"
                print data
                exit(0)
        else:
            cond_num_truth_var = Variable(data)
        loss += self.CE(cond_num_score, cond_num_truth_var)

        # Evaluate the columns of conditions
        T = len(cond_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][4]) > 0:
                truth_prob[b][list(truth_num[b][4])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            cond_col_truth_var = Variable(data.cuda())
        else:
            cond_col_truth_var = Variable(data)

        sigm = nn.Sigmoid()
        cond_col_prob = sigm(cond_col_score)
        bce_loss = -torch.mean(
            3 * (cond_col_truth_var * torch.log(cond_col_prob + 1e-10)) +
            (1 - cond_col_truth_var) * torch.log(1 - cond_col_prob + 1e-10))
        loss += bce_loss

        # Evaluate the operator of conditions
        for b in range(len(truth_num)):
            if len(truth_num[b][5]) == 0:
                continue
            data = torch.from_numpy(np.array(truth_num[b][5]))
            if self.gpu:
                cond_op_truth_var = Variable(data.cuda())
            else:
                cond_op_truth_var = Variable(data)
            cond_op_pred = cond_op_score[b, :len(truth_num[b][5])]
            try:
                loss += (self.CE(cond_op_pred, cond_op_truth_var) / len(truth_num))
            except:
                print cond_op_pred
                print cond_op_truth_var
                exit(0)

        # Evaluate the strings of conditions
        for b in range(len(gt_where)):
            #print(gt_where[b])
	    for idx in range(len(gt_where[b])):
                cond_str_truth = gt_where[b][idx]
		#print("{}{}".format(cond_str_truth, len(cond_str_truth)))
                if len(cond_str_truth) == 2:
		    continue
                #print(cond_str_truth[1:])
		data = torch.from_numpy(np.array(cond_str_truth[1:]))
                #print("data:{}{}".format(data, data.shape))
		if self.gpu:
                    cond_str_truth_var = Variable(data.cuda())
                else:
                    cond_str_truth_var = Variable(data)
                str_end = len(cond_str_truth) - 1
                cond_str_pred = cond_str_score[b, idx, :str_end]
                #print ("cond_str_score:{}cond_str_pred:{}cond_str_truth_var:{}".format(cond_str_score.shape, cond_str_pred.shape, cond_str_truth_var))
		loss += (self.CE(cond_str_pred, cond_str_truth_var) \
                         / (len(gt_where) * len(gt_where[b])))

        # Evaluate condition relationship, and / or
        where_rela_truth = map(lambda x: x[6], truth_num)
        data = torch.from_numpy(np.array(where_rela_truth))
        if self.gpu:
            try:
                where_rela_truth = Variable(data.cuda())
            except:
                print "where_rela_truth error"
                print data
                exit(0)
        else:
            where_rela_truth = Variable(data)
        loss += self.CE(where_rela_score, where_rela_truth)
        return loss
    # def loss(self, score, truth_num, pred_entry, gt_where): #edited by qwy
    #     pred_agg, pred_sel, pred_cond = pred_entry
    #     sel_num_score, agg_score, sel_cond_score, cond_op_str_score, where_rela_score = score
    #
    #     cond_num_score, sel_score, cond_col_score = sel_cond_score
    #     cond_op_score, cond_str_score = cond_op_str_score
    #
    #     loss = 0
    #
    #     sel_num_truth = map(lambda x:x[0], truth_num)
    #     sel_num_truth = torch.from_numpy(np.array(sel_num_truth))
    #     if self.gpu:
    #         sel_num_truth = Variable(sel_num_truth.cuda())
    #     else:
    #         sel_num_truth = Variable(sel_num_truth)
    #     loss += self.CE(sel_num_score, sel_num_truth)
    #
    #     if pred_sel:
    #         sel_truth = map(lambda x:x[1], truth_num)
    #         data = torch.from_numpy(np.array(sel_truth))
    #         if self.gpu:
    #             sel_truth_var = Variable(data.cuda())
    #         else:
    #             sel_truth_var = Variable(data)
    #
    #         loss += self.CE(sel_score, sel_truth_var)
    #
    #     if pred_agg:
    #         agg_truth = map(lambda x:x[2], truth_num)
    #         data = torch.from_numpy(np.array(agg_truth))
    #         if self.gpu:
    #             agg_truth_var = Variable(data.cuda())
    #         else:
    #             agg_truth_var = Variable(data)
    #
    #         loss += self.CE(agg_score, agg_truth_var)
    #
    #     if pred_cond:
    #         B = len(truth_num)
    #         #Evaluate the number of conditions
    #         cond_num_truth = map(lambda x:x[3], truth_num)
    #         data = torch.from_numpy(np.array(cond_num_truth))
    #         if self.gpu:
    #             cond_num_truth_var = Variable(data.cuda())
    #         else:
    #             cond_num_truth_var = Variable(data)
    #         loss += self.CE(cond_num_score, cond_num_truth_var)
    #
    #         #Evaluate the columns of conditions
    #         T = len(cond_col_score[0])
    #         truth_prob = np.zeros((B, T), dtype=np.float32)
    #         for b in range(B):
    #             if len(truth_num[b][4]) > 0:
    #                 truth_prob[b][list(truth_num[b][4])] = 1
    #         data = torch.from_numpy(truth_prob)
    #         if self.gpu:
    #             cond_col_truth_var = Variable(data.cuda())
    #         else:
    #             cond_col_truth_var = Variable(data)
    #
    #         sigm = nn.Sigmoid()
    #         cond_col_prob = sigm(cond_col_score)
    #         bce_loss = -torch.mean( 3*(cond_col_truth_var * \
    #                 torch.log(cond_col_prob+1e-10)) + \
    #                 (1-cond_col_truth_var) * torch.log(1-cond_col_prob+1e-10) )
    #         loss += bce_loss
    #
    #         #Evaluate the operator of conditions
    #         for b in range(len(truth_num)):
    #             if len(truth_num[b][5]) == 0:
    #                 continue
    #             data = torch.from_numpy(np.array(truth_num[b][5]))
    #             if self.gpu:
    #                 cond_op_truth_var = Variable(data.cuda())
    #             else:
    #                 cond_op_truth_var = Variable(data)
    #             cond_op_pred = cond_op_score[b, :len(truth_num[b][5])]
    #             loss += (self.CE(cond_op_pred, cond_op_truth_var) \
    #                     / len(truth_num))
    #
    #         #Evaluate the strings of conditions
    #         for b in range(len(gt_where)):
    #             for idx in range(len(gt_where[b])):
    #                 cond_str_truth = gt_where[b][idx]
    #                 if len(cond_str_truth) == 1:
    #                     continue
    #                 data = torch.from_numpy(np.array(cond_str_truth[1:]))
    #                 if self.gpu:
    #                     cond_str_truth_var = Variable(data.cuda())
    #                 else:
    #                     cond_str_truth_var = Variable(data)
    #                 str_end = len(cond_str_truth)-1
    #                 cond_str_pred = cond_str_score[b, idx, :str_end]
    #                 loss += (self.CE(cond_str_pred, cond_str_truth_var) \
    #                         / (len(gt_where) * len(gt_where[b])))
    #
    #         where_rela_truth = map(lambda x: x[6], truth_num)
    #         data = torch.from_numpy(np.array(where_rela_truth))
    #         if self.gpu:
    #             try:
    #                 where_rela_truth = Variable(data.cuda())
    #             except:
    #                 print "where_rela_truth error"
    #                 print data
    #                 exit(0)
    #         else:
    #             where_rela_truth = Variable(data)
    #         loss += self.CE(where_rela_score, where_rela_truth)
    #
    #     return loss


    def check_acc(self, vis_info, pred_queries, gt_queries):
        def pretty_print(vis_data, pred_query, gt_query):
            print "\n----------detailed error prints-----------"
            try:
                print 'question: ', vis_data[0]
                print 'question_tok: ', vis_data[3]
                print 'headers: (%s)'%(' || '.join(vis_data[1]))
                print 'query:', vis_data[2]
                print "target query: ", gt_query
                print "pred query: ", pred_query
            except:
                print "\n------skipping print: decoding problem ----------------------"

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(header[cond[0]] + ' ' +
                    self.COND_OPS[cond[1]] + ' ' + unicode(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        """
	B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = 0.0
        cond_num_err = cond_col_err = cond_op_err = cond_val_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        for b, (pred_qry, gt_qry, vis_data) in enumerate(zip(pred_queries, gt_queries, vis_info)):
            good = True
            if pred_agg:
                agg_pred = pred_qry['agg']
                agg_gt = gt_qry['agg']
                if agg_pred != agg_gt:
                    agg_err += 1
                    good = False

            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']
                if sel_pred != sel_gt:
                    sel_err += 1
                    good = False

            if pred_cond:
                cond_pred = pred_qry['conds']
                cond_gt = gt_qry['conds']
                flag = True
                if len(cond_pred) != len(cond_gt):
                    flag = False
                    cond_num_err += 1

                if flag and set(x[0] for x in cond_pred) != \
                        set(x[0] for x in cond_gt):
                    flag = False
                    cond_col_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                        flag = False
                        cond_op_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(
                            x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and unicode(cond_gt[gt_idx][2]).lower() != \
                            unicode(cond_pred[idx][2]).lower():
                        flag = False
                        cond_val_err += 1

                if not flag:
                    cond_err += 1
                    good = False

            if not good:
                if error_print:
                    pretty_print(vis_data, pred_qry, gt_qry)
                tot_err += 1

        return np.array((agg_err, sel_err, cond_err, cond_num_err, cond_col_err, cond_op_err, cond_val_err)), tot_err
"""
        tot_err = sel_num_err = agg_err = sel_err = 0.0
        cond_num_err = cond_col_err = cond_op_err = cond_val_err = cond_rela_err = 0.0
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            sel_pred, agg_pred, where_rela_pred = pred_qry['sel'], pred_qry['agg'], pred_qry['cond_conn_op']
            sel_gt, agg_gt, where_rela_gt = gt_qry['sel'], gt_qry['agg'], gt_qry['cond_conn_op']

            if where_rela_gt != where_rela_pred:
                good = False
                cond_rela_err += 1

            if len(sel_pred) != len(sel_gt):
                good = False
                sel_num_err += 1

            pred_sel_dict = {k:v for k,v in zip(list(sel_pred), list(agg_pred))}
            gt_sel_dict = {k:v for k,v in zip(sel_gt, agg_gt)}
            if set(sel_pred) != set(sel_gt):
                good = False
                sel_err += 1
            agg_pred = [pred_sel_dict[x] for x in sorted(pred_sel_dict.keys())]
            agg_gt = [gt_sel_dict[x] for x in sorted(gt_sel_dict.keys())]
            if agg_pred != agg_gt:
                good = False
                agg_err += 1

            cond_pred = pred_qry['conds']
            cond_gt = gt_qry['conds']
            if len(cond_pred) != len(cond_gt):
                good = False
                cond_num_err += 1
            else:
                cond_op_pred, cond_op_gt = {}, {}
                cond_val_pred, cond_val_gt = {}, {}
                for p, g in zip(cond_pred, cond_gt):
                    cond_op_pred[p[0]] = p[1]
                    cond_val_pred[p[0]] = p[2]
                    cond_op_gt[g[0]] = g[1]
                    cond_val_gt[g[0]] = g[2]

                if set(cond_op_pred.keys()) != set(cond_op_gt.keys()):
                    cond_col_err += 1
                    good=False

                where_op_pred = [cond_op_pred[x] for x in sorted(cond_op_pred.keys())]
                where_op_gt = [cond_op_gt[x] for x in sorted(cond_op_gt.keys())]
                if where_op_pred != where_op_gt:
                    cond_op_err += 1
                    good=False

                where_val_pred = [cond_val_pred[x] for x in sorted(cond_val_pred.keys())]
                where_val_gt = [cond_val_gt[x] for x in sorted(cond_val_gt.keys())]
                if where_val_pred != where_val_gt:
                    cond_val_err += 1
                    good=False

            if not good:
                tot_err += 1

        return np.array((sel_num_err, sel_err, agg_err, cond_num_err, cond_col_err, cond_op_err, cond_val_err , cond_rela_err)), tot_err

    def gen_query(self, score, q, col, raw_q, reinforce=False, verbose=False):
        def merge_tokens(tok_list, raw_tok_str):
            """
            tok_list: list of string words in current cond
            raw_tok_str: list of words in question
            """
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-':'(',
                    '-RRB-':')',
                    '-LSB-':'[',
                    '-RSB-':']',
                    '``':'"',
                    '\'\'':'"',
                    '--':u'\u2013'}
            ret = ''
            double_quote_appear = 0
            tok_list = [x for gx in tok_list for x in gx]
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
                        and (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        sel_cond_score, agg_score, cond_op_str_score, where_rela_score = score

        sel_num_score, cond_num_score, sel_score, cond_col_score \
            = [x.data.cpu().numpy() for x in sel_cond_score]
        cond_op_score, cond_str_score = [x.data.cpu().numpy() for x in cond_op_str_score]

        # [64,4,6], [64,14], ..., [64,4]
        #sel_num_score = sel_num_score.data.cpu().numpy()
        #sel_score = sel_score.data.cpu().numpy()
        agg_score = agg_score.data.cpu().numpy()
        where_rela_score = where_rela_score.data.cpu().numpy()
        ret_queries = []
        B = len(agg_score)

        for b in range(B):
            cur_query = {}
            cur_query['sel'] = []
            cur_query['agg'] = []
            sel_num = np.argmax(sel_num_score[b])
            max_col_idxes = np.argsort(-sel_score[b])[:sel_num]
            # find the most-probable columns' indexes
            max_agg_idxes = np.argsort(-agg_score[b])[:sel_num]
            cur_query['sel'].extend([int(i) for i in max_col_idxes])
            cur_query['agg'].extend([i[0] for i in max_agg_idxes])
            cur_query['cond_conn_op'] = np.argmax(where_rela_score[b])
            cur_query['conds'] = []
            cond_num = np.argmax(cond_num_score[b])
            all_toks = ['<BEG>'] + q[b] + ['<END>']
            max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
            for idx in range(cond_num):
                cur_cond = []
                cur_cond.append(max_idxes[idx]) # where-col
                cur_cond.append(np.argmax(cond_op_score[b][idx])) # where-op
                cur_cond_str_toks = []
                for str_score in cond_str_score[b][idx]:
                    str_tok = np.argmax(str_score[:len(all_toks)])
                    str_val = all_toks[str_tok]
                    if str_val == '<END>':
                        break
                    cur_cond_str_toks.append(str_val)
                cur_cond.append(merge_tokens(cur_cond_str_toks, raw_q[b]))
                cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)
        return ret_queries

