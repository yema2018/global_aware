from train_att_prediction import TrainPreAtt, positional_encoding
from transformers import BartForConditionalGeneration, BartTokenizer, MBartForConditionalGeneration, MBartTokenizer
from generate_batch import gen_bt
import torch
import argparse
from att_pred_model import PreAttModel
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Attention-aware Inference')
    parser.add_argument('--ckpt', nargs='?', default='', help='checkpoint path')
    parser.add_argument('--dataset', nargs='?', default='', help='cnndm or xsum')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', type=int, default=4, help='epoch')

    parser.add_argument('--beam_size', type=int, default=4, help='beam search size.')
    parser.add_argument('--beta', type=float, default=8, help='the coefficient of att-aware')
    parser.add_argument('--gamma', type=float, default=1, help='the coefficient of length penalty in att-aware')
    parser.add_argument('--ada', type=float, default=30, help='thresold')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help="The parameter for repetition penalty. 1.0 means no penalty. See `this paper"
                             " <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.")
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0,
                        help='If set to int > 0, all ngrams of that size can only occur once.')

    parser.add_argument('--cuda', type=int, default=0,
                        help='the index of cuda used.')

    parser.add_argument('--trunc', type=int, default=1024,
                        help='truncate inp of att-pred-model into this length.')

    parser.add_argument('--train', dest='train', action='store_true', help='training.')
    parser.add_argument('--att_aware', dest='att_aware', action='store_true',
                        help='enter attention-aware inference.')
    parser.add_argument('--vanilla', dest='vanilla', action='store_true', help='enter vanilla beam search.')
    parser.add_argument('--cheat', dest='cheat', action='store_true', help='enter cheating att-aware inference.')
    parser.add_argument('--cross', dest='cross', action='store_true', help='enter cross att-aware inference.')

    return parser.parse_args()


args = parse_args()


def inference(summ, tokenizer):
    ckpt = args.ckpt
    device = torch.device('cuda: {}'.format(args.cuda))
    if args.att_aware:
        print('enter attention-aware inference.')
        pre_att_model =PreAttModel(layers=5, d_model=1024, num_heads=16, dff=4096, rate=0.1)

        try:
            pre_att_model.load_state_dict(torch.load(ckpt))
            print('load {}'.format(ckpt))
        except:
            print('no checkpoints now!')

        pre_att_model.eval()
        pre_att_model.to(device)

        encoder = summ.get_encoder()
        encoder.eval()
        encoder.to(device)

    if args.vanilla:
        print('enter vanilla beam search.')

    if args.cheat:
        print('enter cheating att-aware inference.')

    if args.cross and not args.vanilla:
        print('enter cross att-aware inference.')
        pre_att_model =PreAttModel(layers=5, d_model=1024, num_heads=16, dff=4096, rate=0.1)

        # pos_ec = positional_encoding(2000, 1024)
        try:
            pre_att_model.load_state_dict(torch.load(ckpt))
            print('load {}'.format(ckpt))
        except:
            print('no checkpoints now!')

        pre_att_model.eval()
        pre_att_model.to(device)

        encoder = summ.get_encoder()
        encoder.eval()
        encoder.to(device)
    summ.eval()

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     summ = torch.nn.DataParallel(summ)

    summ.to(device)
    batch_set = gen_bt(1, tokenizer, 'test', dataset=args.dataset)  # the batch_size can not > 1
    start = time.time()
    for (batch, batch_contents) in enumerate(batch_set):
        inp, tar, inp_mask, tar_mask = batch_contents
        inp = inp.to(device)
        inp_mask = inp_mask.to(device)
        if args.cheat:
            tar = tar.to(device)
            tar_mask = tar_mask.to(device)
        with torch.no_grad():
            if args.att_aware:
                encoder_out = encoder(inp, inp_mask, return_dict=True).last_hidden_state
                opt = pre_att_model(encoder_out[:, :args.trunc, :], inp_mask[:, :args.trunc])
                opt_len = torch.sum(opt)
                gamma = np.log((np.max([opt_len.item(), args.ada+1])-args.ada)**0.5)
                summary_ids, _ = summ.generate(inp, attention_mask=inp_mask, num_beams=args.beam_size,
                                            early_stopping=True, opt_att_dist=opt, min_length=0, length_penalty=1.0,
                                            beta=args.beta, repetition_penalty=args.repetition_penalty,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size, max_length=152, gamma=gamma)
                out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]
                print(out_list)

                for i in out_list:
                    with open('{}/attADA{}_beta{}_beam{}_ga{}.txt'.format(args.dataset, args.ada, args.beta, args.beam_size, args.gamma),
                              'a', encoding='utf8') as fw:
                        fw.write(' .'.join(i.split('.')))
                        fw.write('\n')
            if args.cross:
                if args.vanilla:
                    opt = None
                    prefix = 'vanilla_new'
                else:
                    encoder_out = encoder(inp, inp_mask, return_dict=True).last_hidden_state
                    opt = pre_att_model(encoder_out[:, :args.trunc, :], inp_mask[:, :args.trunc])
                    prefix = 'cross'
                summary_ids, _ = summ.generate(inp, attention_mask=inp_mask, num_beams=args.beam_size,
                                               early_stopping=True, opt_att_dist=opt, min_length=0, length_penalty=1.0,
                                               beta=args.beta, repetition_penalty=args.repetition_penalty,
                                               no_repeat_ngram_size=args.no_repeat_ngram_size, max_length=152, gamma=args.gamma)
                out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]
                print(out_list)

                for i in out_list:
                    with open(
                            '{}/{}_beam{}.txt'.format(args.dataset, prefix, args.beam_size),
                            'a', encoding='utf8') as fw:
                        fw.write(' .'.join(i.split('.')))
                        fw.write('\n')
            if args.vanilla and not args.cross:
                summary_ids, _ = bart.generate(inp, attention_mask=inp_mask, num_beams=args.beam_size,
                                            early_stopping=True, opt_att_dist=None,
                                            repetition_penalty=args.repetition_penalty,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size)
                out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]
                print(out_list)
                for i in out_list:
                    with open('{}/vanilla_beam{}_rp{}_nr{}.txt'.format(args.dataset, args.beam_size, args.repetition_penalty,
                                                                          args.no_repeat_ngram_size),
                              'a', encoding='utf8') as fw:
                        fw.write(' .'.join(i.split('.')))
                        fw.write('\n')

            if args.cheat:
                summ_output = summ(input_ids=inp, attention_mask=inp_mask, decoder_input_ids=tar[:, :-1],
                                   output_attentions=True, output_hidden_states=True, return_dict=True)

                decoder_att = summ_output.decoder_attentions
                opt_att_dist = 0
                for _ in decoder_att:
                    opt_att_dist += _[:, :, :, -inp.size()[1]:]
                opt_att_dist = torch.sum(torch.mean(opt_att_dist, dim=1), dim=1)
                opt_att_dist /= len(decoder_att)
                opt_len = torch.sum(opt_att_dist)
                gamma = np.log((np.max([opt_len.item(), args.ada+1]) - args.ada) ** 0.5)
                summary_ids, _ = summ.generate(inp, attention_mask=inp_mask, num_beams=args.beam_size,
                                            early_stopping=True, opt_att_dist=opt_att_dist,min_length=0,length_penalty=1.0,
                                            beta=args.beta, repetition_penalty=args.repetition_penalty,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size, max_length=152, gamma=gamma)
                out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]
                print(out_list)
                for i in out_list:
                    with open('{}/cheatADA{}_beta{}_beam{}_ga{}.txt'.format(args.dataset, args.ada, args.beta, args.beam_size, args.gamma),
                              'a', encoding='utf8') as fw:
                        fw.write(' .'.join(i.split('.')))
                        fw.write('\n')

    print('inference_time: {}'.format(time.time()-start))


if __name__ == '__main__':
    assert args.dataset in ['cnndm','xsum','wmt']
    if args.dataset == 'cnndm':
        path = '/root/yema/bart-large-cnn'
        bart = BartForConditionalGeneration.from_pretrained(path, use_cache=False)
        tokenizer = BartTokenizer.from_pretrained('/root/yema/bart-large-vocab')
    if args.dataset == 'xsum':
        path = '/root/yema/bart-large-xsum'
        bart = BartForConditionalGeneration.from_pretrained(path, use_cache=False)
        tokenizer = BartTokenizer.from_pretrained('/root/yema/bart-large-vocab')
    if args.dataset == 'wmt':
        path = '/root/yema/mbart-large-en-ro'
        bart = MBartForConditionalGeneration.from_pretrained(path, use_cache=False)
        tokenizer = MBartTokenizer.from_pretrained('/root/yema/mbart-large-vocab')

    if args.train:
        a = TrainPreAtt(bart, tokenizer, args.ckpt, args.epoch, args.batch_size, args.dataset)
        a.train()
    else:
        inference(bart, tokenizer)
