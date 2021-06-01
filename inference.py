from train_att_prediction import TrainPreAtt
from transformers import BartForConditionalGeneration, BartTokenizer, PegasusForConditionalGeneration, PegasusTokenizer
from generate_batch import gen_bt
import torch
import argparse
from att_pred_model import PreAttModel
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Global-aware Inference')
    parser.add_argument('--ckpt', nargs='?', default='', help='checkpoint path')
    parser.add_argument('--dataset', nargs='?', default='', help='cnndm or xsum')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', type=int, default=4, help='epoch')

    parser.add_argument('--beam_size', type=int, default=4, help='beam search size.')
    parser.add_argument('--beta', type=float, default=12, help='the coefficient of global-aware')
    parser.add_argument('--gamma', type=float, default=1, help='the coefficient of length penalty in global-aware')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help="The parameter for repetition penalty. 1.0 means no penalty. See `this paper"
                             " <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.")
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0,
                        help='If set to int > 0, all ngrams of that size can only occur once.')

    parser.add_argument('--cuda', type=int, default=0,
                        help='the index of cuda used.')

    parser.add_argument('--train', dest='train', action='store_true', help='training.')
    parser.add_argument('--global_aware', dest='global_aware', action='store_true',
                        help='enter global-aware inference.')
    parser.add_argument('--vanilla', dest='vanilla', action='store_true', help='enter vanilla beam search.')
    parser.add_argument('--oracle', dest='oracle', action='store_true', help='enter oracle global-aware inference.')
    parser.add_argument('--vanilla_no', dest='vanilla_no', action='store_true', help='enter vanilla beam search without'
                                                                                     'length limits.')
    parser.add_argument('--peg', dest='peg', action='store_true', help='use pegasus')

    return parser.parse_args()


args = parse_args()


def inference(summ, tokenizer, summ_use):
    ckpt = args.ckpt
    model_fix = 'bart'
    ml = 1024
    if args.peg:
        model_fix = 'peg'
        if args.dataset in ['xsum', 'newsroom','wikihow','reddit']:
            ml = 512
    device = torch.device('cuda: {}'.format(args.cuda))

    if args.global_aware:
        print('enter global-aware inference.')
        pre_att_model =PreAttModel(layers=2, d_model=1024, num_heads=16, dff=4096, rate=0.0)

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

    if args.oracle:
        print('enter oracle global-aware inference.')

    if args.vanilla_no:
        print('enter vanilla beam search without length limits.')

    summ.eval()

    summ.to(device)
    batch_set = gen_bt(1, tokenizer, 'test', dataset=args.dataset, ml=ml, peg=args.peg)  # the batch_size can not > 1
    start = time.time()

    if summ_use is None:
        summ_g = summ
    else:
        summ_use.to(device)
        summ_g = summ_use

    for (batch, batch_contents) in enumerate(batch_set):
        inp, tar, inp_mask, tar_mask = batch_contents
        inp = inp.to(device)
        inp_mask = inp_mask.to(device)
        if args.oracle:
            tar = tar.to(device)
            tar_mask = tar_mask.to(device)
        with torch.no_grad():
            if args.global_aware:
                encoder_out = encoder(inp, inp_mask, return_dict=True).last_hidden_state
                opt = pre_att_model(encoder_out, inp_mask)

                summary_ids, _ = summ_g.generate(inp, attention_mask=inp_mask, num_beams=args.beam_size,
                                            early_stopping=True, opt_att_dist=opt, min_length=1, length_penalty=1.0,
                                            beta=args.beta, repetition_penalty=args.repetition_penalty,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size, max_length=256, gamma=args.gamma)

                out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]
                print(out_list)

                for i in out_list:
                    with open('{}/global_{}_beta{}_beam{}_ga{}.txt'.format(args.dataset,model_fix, args.beta, args.beam_size, args.gamma),
                              'a', encoding='utf8') as fw:
                        fw.write(' .'.join(i.split('.')).replace('<n>', ' '))
                        fw.write('\n')
            if args.vanilla_no:
                summary_ids, _ = summ_g.generate(inp, attention_mask=inp_mask, num_beams=args.beam_size,
                                               early_stopping=True, opt_att_dist=None, min_length=1, length_penalty=1.0,
                                               repetition_penalty=args.repetition_penalty,
                                               no_repeat_ngram_size=args.no_repeat_ngram_size, max_length=256)

                out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]
                print(out_list)

                for i in out_list:
                    with open(
                            '{}/vanilla_no_beam{}.txt'.format(args.dataset, args.beam_size),
                            'a', encoding='utf8') as fw:
                        fw.write(' .'.join(i.split('.')).replace('<n>', ' '))
                        fw.write('\n')
            if args.vanilla:
                summary_ids, _ = summ_g.generate(inp, attention_mask=inp_mask, num_beams=args.beam_size,
                                            early_stopping=True, opt_att_dist=None,
                                            repetition_penalty=args.repetition_penalty,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size)
                out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]
                print(out_list)

                for i in out_list:
                    with open('{}/vanilla_{}_beam{}.txt'.format(args.dataset, model_fix, args.beam_size),
                              'a', encoding='utf8') as fw:
                        fw.write(' .'.join(i.split('.')).replace('<n>', ' '))
                        fw.write('\n')

            if args.oracle:
                summ_output = summ(input_ids=inp, attention_mask=inp_mask, decoder_input_ids=tar[:, :-1],
                                   output_attentions=True, output_hidden_states=True, return_dict=True)

                decoder_att = summ_output.decoder_attentions
                opt_att_dist = 0
                for _ in decoder_att:
                    opt_att_dist += _[:, :, :, -inp.size()[1]:]
                opt_att_dist = torch.sum(torch.mean(opt_att_dist, dim=1), dim=1)
                opt_att_dist /= len(decoder_att)

                summary_ids, _ = summ_g.generate(inp, attention_mask=inp_mask, num_beams=args.beam_size,
                                            early_stopping=True, opt_att_dist=opt_att_dist, min_length=1, length_penalty=1.0,
                                            beta=args.beta, repetition_penalty=args.repetition_penalty,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size, max_length=256, gamma=args.gamma)

                out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]
                print(out_list)
                for i in out_list:
                    with open('{}/oracle_{}_beta{}_beam{}_ga{}.txt'.format(args.dataset, model_fix, args.beta, args.beam_size, args.gamma),
                              'a', encoding='utf8') as fw:
                        fw.write(' .'.join(i.split('.')).replace('<n>', ' '))
                        fw.write('\n')

    print('inference_time: {}'.format(time.time()-start))


if __name__ == '__main__':
    assert args.dataset in ['cnndm','xsum','newsroom','multi-news','billsum','reddit','wikihow','arxiv']
    summ_use = None
    if args.dataset == 'cnndm':
        if not args.peg:
            path = 'facebook/bart-large-cnn'
            summ = BartForConditionalGeneration.from_pretrained(path, use_cache=False)
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    if args.dataset == 'xsum':
        if not args.peg:
            path = 'facebook/bart-large-xsum'
            summ = BartForConditionalGeneration.from_pretrained(path, use_cache=False)
            summ_use = BartForConditionalGeneration.from_pretrained(path)
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
        else:
            path = 'google/pegasus-xsum'
            summ = PegasusForConditionalGeneration.from_pretrained(path, use_cache=False)
            if not args.train:
                summ_use = PegasusForConditionalGeneration.from_pretrained(path)
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    if args.dataset == 'newsroom':
        if not args.peg:
            path = 'facebook/bart-large-cnn'
            summ = BartForConditionalGeneration.from_pretrained(path, use_cache=False)
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        else:
            path = 'google/pegasus-newsroom'
            summ = PegasusForConditionalGeneration.from_pretrained(path, use_cache=False)
            if not args.train:
                summ_use = PegasusForConditionalGeneration.from_pretrained(path)
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-newsroom')
    if args.dataset == 'multi-news':
        if args.peg:
            path = 'google/pegasus-multi_news'
            summ = PegasusForConditionalGeneration.from_pretrained(path, use_cache=False)
            if not args.train:
                summ_use = PegasusForConditionalGeneration.from_pretrained(path)
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-multi-news')
    if args.dataset == 'billsum':
        if args.peg:
            path = 'google/pegasus-billsum'
            summ = PegasusForConditionalGeneration.from_pretrained(path, use_cache=False)
            if not args.train:
                summ_use = PegasusForConditionalGeneration.from_pretrained(path)
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-billsum')
    if args.dataset == 'reddit':
        if args.peg:
            path = 'google/pegasus-reddit_tifu'
            summ = PegasusForConditionalGeneration.from_pretrained(path, use_cache=False)
            if not args.train:
                summ_use = PegasusForConditionalGeneration.from_pretrained(path)
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-reddit_tifu')
    if args.dataset == 'wikihow':
        if args.peg:
            path = 'google/pegasus-wikihow'
            summ = PegasusForConditionalGeneration.from_pretrained(path, use_cache=False)
            if not args.train:
                summ_use = PegasusForConditionalGeneration.from_pretrained(path)
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-wikihow')
    if args.dataset == 'arxiv':
        if args.peg:
            path = 'google/pegasus-arxiv'
            summ = PegasusForConditionalGeneration.from_pretrained(path, use_cache=False)
            if not args.train:
                summ_use = PegasusForConditionalGeneration.from_pretrained(path)
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-arxiv')


    if args.train:
        a = TrainPreAtt(summ, tokenizer, args.ckpt, args.epoch, args.batch_size, args.dataset, args.peg)
        a.train()
    else:
        inference(summ, tokenizer, summ_use)



