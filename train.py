import json
import torch
import datetime
import argparse
import numpy as np
from typesql.utils import *
from typesql.model.sqlnet import SQLNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch number')
    parser.add_argument('--gpu', action='store_true', help='Whether use gpu to train')
    parser.add_argument('--toy', action='store_true', help='If set, use small data for fast debugging')
    parser.add_argument('--ca', action='store_true', help='Whether use column attention')
    parser.add_argument('--train_emb', action='store_true', help='Train word embedding for SQLNet')
    parser.add_argument('--restore', action='store_true', help='Whether restore trained model')
    parser.add_argument('--logdir', type=str, default='', help='Path of save experiment log')
    parser.add_argument('--db_content', type=int, default=0,
            help='0: use knowledge graph type, 1: use db content to get type info')

    args = parser.parse_args()

    n_word = 600
    if args.toy:
        use_small = True
        gpu = args.gpu
        batch_size = 2 
    else:
        use_small = False
        gpu = args.gpu
        batch_size = args.bs
    learning_rate = 1e-3

    # load dataset
    train_sql, train_table, train_db, dev_sql, dev_table, dev_db = load_dataset(use_small=use_small)

    #word_emb = load_word_emb('data_zhuiyi/sgns.baidubaike.bigram-char')
    word_emb = load_concat_wemb('data_zhuiyi/sgns.baidubaike.bigram-char', 'data_zhuiyi/hanlp-wiki-vec-zh')
    #word_emb = load_concat_wemb('data_zhuiyi/char_embedding', 'data_zhuiyi/char_embedding')
    model = SQLNet(word_emb, N_word=n_word, use_ca=args.ca, gpu=gpu, trainable_emb=args.train_emb, db_content=args.db_content)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    if args.restore:
        model_path = 'saved_model/best_model'
        print "Loading trained model from %s" % model_path
        model.load_state_dict(torch.load(model_path))

    best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv, best_wr = 0, 0, 0, 0, 0, 0, 0, 0
    best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx, best_wr_idx = 0, 0, 0, 0, 0, 0, 0, 0
    best_lf, best_lf_idx = 0.0, 0
    best_ex, best_ex_idx = 0.0, 0

    print "#" * 20 + "  Star to Train  " + "#" * 20
    for i in range(args.epoch):
        print 'Epoch %d' % (i + 1)
        # train on the train dataset
        train_loss = epoch_train(model, optimizer, batch_size, train_sql, train_table, args.db_content)
        # evaluate on the dev dataset
        dev_acc = epoch_acc(model, batch_size, dev_sql, dev_table, dev_db, args.db_content)
        # accuracy of each sub-task
        print 'Sel-Num: %.3f, Sel-Col: %.3f, Sel-Agg: %.3f, W-Num: %.3f, W-Col: %.3f, W-Op: %.3f, W-Val: %.3f, W-Rel: %.3f' % (
            dev_acc[0][0], dev_acc[0][1], dev_acc[0][2], dev_acc[0][3], dev_acc[0][4], dev_acc[0][5], dev_acc[0][6],
            dev_acc[0][7])
        # save the best model
        if dev_acc[1] > best_lf:
            best_lf = dev_acc[1]
            best_lf_idx = i + 1
            torch.save(model.state_dict(), 'saved_model/best_model')
        if dev_acc[2] > best_ex:
            best_ex = dev_acc[2]
            best_ex_idx = i + 1

        # record the best score of each sub-task
        if True:
            if dev_acc[0][0] > best_sn:
                best_sn = dev_acc[0][0]
                best_sn_idx = i + 1
            if dev_acc[0][1] > best_sc:
                best_sc = dev_acc[0][1]
                best_sc_idx = i + 1
                best_sc_idx = i + 1
            if dev_acc[0][2] > best_sa:
                best_sa = dev_acc[0][2]
                best_sa_idx = i + 1
            if dev_acc[0][3] > best_wn:
                best_wn = dev_acc[0][3]
                best_wn_idx = i + 1
            if dev_acc[0][4] > best_wc:
                best_wc = dev_acc[0][4]
                best_wc_idx = i + 1
            if dev_acc[0][5] > best_wo:
                best_wo = dev_acc[0][5]
                best_wo_idx = i + 1
            if dev_acc[0][6] > best_wv:
                best_wv = dev_acc[0][6]
                best_wv_idx = i + 1
            if dev_acc[0][7] > best_wr:
                best_wr = dev_acc[0][7]
                best_wr_idx = i + 1
        print 'Train loss = %.3f' % train_loss
        print 'Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2])
        print 'Best Logic Form: %.3f at epoch %d' % (best_lf, best_lf_idx)
        print 'Best Execution: %.3f at epoch %d' % (best_ex, best_ex_idx)
        if (i + 1) % 10 == 0:
            print 'Best val acc: %s\nOn epoch individually %s' % (
                (best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv),
                (best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx))

    # N_word=300
    # B_word=42
    # if args.toy:
    #     USE_SMALL=True
    #     GPU=True
    #     BATCH_SIZE=15
    # else:
    #     USE_SMALL=False
    #     GPU=True
    #     BATCH_SIZE=64
    # TRAIN_ENTRY=(True, True, True, True, True)  # (AGG, SEL_NUM, SEL, COND, WHERE_RELA)
    # TRAIN_SEL_NUM, TRAIN_AGG, TRAIN_SEL, TRAIN_COND, TRAIN_WHERE_RELA = TRAIN_ENTRY
    # learning_rate = 1e-3
    #
    # # sql_data, table_data, val_sql_data, val_table_data, \
    # #         test_sql_data, test_table_data, \
    # #         TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)
    #
    # # load dataset
    # train_sql, train_table, train_db, dev_sql, dev_table, dev_db = load_dataset(use_small=USE_SMALL)
    # #word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
    # #        load_used=args.train_emb, use_small=USE_SMALL)
    # if args.db_content == 0:
    #     word_emb = load_word_and_type_emb('glove/glove.42B.300d.txt', "para-nmt-50m/data/paragram_sl999_czeng.txt",\
    #                                         dev_sql, dev_table, args.db_content, is_list=True, use_htype=False)
    # else:
    #     # word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "para-nmt-50m/data/paragram_sl999_czeng.txt")
    #     word_emb = load_word_emb('data_zhuiyi/sgns.baidubaike.bigram-char')
    #
    # model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)
    #
    # agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)
    #
    # if args.train_emb: # Load pretrained model.
    #     agg_lm, sel_lm, cond_lm = best_model_name(args, for_load=True)
    #     print "Loading from %s"%agg_lm
    #     model.agg_pred.load_state_dict(torch.load(agg_lm))
    #     print "Loading from %s"%sel_lm
    #     model.selcond_pred.load_state_dict(torch.load(sel_lm))
    #     print "Loading from %s"%cond_lm
    #     model.cond_pred.load_state_dict(torch.load(cond_lm))


    # #initial accuracy
    # init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, args.db_content)
    # best_agg_acc = init_acc[1][0]
    # best_agg_idx = 0
    # best_sel_acc = init_acc[1][1]
    # best_sel_idx = 0
    # best_cond_acc = init_acc[1][2]
    # best_cond_idx = 0
    # print 'Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s' % init_acc
    # if TRAIN_AGG:
    #     torch.save(model.agg_pred.state_dict(), agg_m)
    #     torch.save(model.agg_type_embed_layer.state_dict(), agg_e)
    # if TRAIN_SEL:
    #     torch.save(model.selcond_pred.state_dict(), sel_m)
    #     torch.save(model.sel_type_embed_layer.state_dict(), sel_e)
    # if TRAIN_COND:
    #     torch.save(model.op_str_pred.state_dict(), cond_m)
    #     torch.save(model.cond_type_embed_layer.state_dict(), cond_e)
    #
    # for i in range(100):
    #     print 'Epoch %d @ %s'%(i+1, datetime.datetime.now())
    #     print ' Loss = %s'%epoch_train(
    #             model, optimizer, BATCH_SIZE,
    #             sql_data, table_data, TRAIN_ENTRY, args.db_content)
    #     print ' Train acc_qm: %s\n breakdown result: %s'%epoch_acc(
    #             model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY, args.db_content)
    #
    #     val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, args.db_content, False) #for detailed error analysis, pass True to the end
    #     print ' Dev acc_qm: %s\n breakdown result: %s'%val_acc
    #     if TRAIN_AGG:
    #         if val_acc[1][0] > best_agg_acc:
    #             best_agg_acc = val_acc[1][0]
    #             best_agg_idx = i+1
    #             torch.save(model.agg_pred.state_dict(),
    #                 args.sd + '/epoch%d.agg_model%s'%(i+1, args.suffix))
    #             torch.save(model.agg_pred.state_dict(), agg_m)
    #
    #         torch.save(model.agg_type_embed_layer.state_dict(),
    #                             args.sd + '/epoch%d.agg_embed%s'%(i+1, args.suffix))
    #         torch.save(model.agg_type_embed_layer.state_dict(), agg_e)
    #
    #     if TRAIN_SEL:
    #         if val_acc[1][1] > best_sel_acc:
    #             best_sel_acc = val_acc[1][1]
    #             best_sel_idx = i+1
    #             torch.save(model.selcond_pred.state_dict(),
    #                 args.sd + '/epoch%d.sel_model%s'%(i+1, args.suffix))
    #             torch.save(model.selcond_pred.state_dict(), sel_m)
    #
    #             torch.save(model.sel_type_embed_layer.state_dict(),
    #                             args.sd + '/epoch%d.sel_embed%s'%(i+1, args.suffix))
    #             torch.save(model.sel_type_embed_layer.state_dict(), sel_e)
    #
    #     if TRAIN_COND:
    #         if val_acc[1][2] > best_cond_acc:
    #             best_cond_acc = val_acc[1][2]
    #             best_cond_idx = i+1
    #             torch.save(model.op_str_pred.state_dict(),
    #                 args.sd + '/epoch%d.cond_model%s'%(i+1, args.suffix))
    #             torch.save(model.op_str_pred.state_dict(), cond_m)
    #
    #             torch.save(model.cond_type_embed_layer.state_dict(),
    #                             args.sd + '/epoch%d.cond_embed%s'%(i+1, args.suffix))
    #             torch.save(model.cond_type_embed_layer.state_dict(), cond_e)
    #
    #     print ' Best val acc = %s, on epoch %s individually'%(
    #             (best_agg_acc, best_sel_acc, best_cond_acc),
    #             (best_agg_idx, best_sel_idx, best_cond_idx))
