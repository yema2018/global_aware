from att_prediction import PreAtt
from transformers import BartForConditionalGeneration, BartTokenizer
from generate_batch import gen_bt
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Run graph2vec based MDS tasks.')
    parser.add_argument('--mode', nargs='?', default='train', help='must be the train/gen')
    parser.add_argument('--ckpt', nargs='?', default='', help='checkpoint path')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epoch', type=int, default=4, help='epoch')

    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers in att_pred_model')

    parser.add_argument('--beam_size', type=int, default=5, help='beam search size.')
    parser.add_argument('--beta', type=float, default=0.8, help='the coefficient of att-aware')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help="The parameter for repetition penalty. 1.0 means no penalty. See `this paper"
                             " <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.")
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0,
                        help='If set to int > 0, all ngrams of that size can only occur once.')
    parser.add_argument('--length_penalty', type=float, default=1.0,
                        help='Exponential penalty to the length. 1.0 means no penalty.')
    parser.add_argument('--max_length', type=int, default=200,
                        help='Max_length of generated sequences.')

    parser.add_argument('--att_aware', dest='att_aware', action='store_true',
                        help='Boolean specifying using att-aware or vanilla beam search. Default is with att-aware.')
    parser.add_argument('--vanilla', dest='vanilla', action='store_false')
    parser.set_defaults(att_aware=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn', use_cache=False)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    a = PreAtt(bart, tokenizer, args.ckpt, args.epoch, args.batch_size, args.num_layers)
    if args.mode == 'train':
        a.train()
    else:
        encoder = a.summ_encoder()
        device = torch.device('cuda')
        batch_set = gen_bt(args.batch_size, tokenizer, 'test')
        for (batch, batch_contents) in enumerate(batch_set):
            inp, _, inp_mask, _ = batch_contents
            inp = inp.to(device)
            inp_mask = inp_mask.to(device)
            if args.att_aware:
                encoder_out = encoder(inp, inp_mask, return_dict=True).last_hidden_state
                opt = a.ex_pred(encoder_out, inp_mask)
                summary_ids = bart.generate(inp, attention_mask=inp_mask, num_beams=args.beam_size,
                                            max_length=args.max_length, early_stopping=True, opt_att_dist=opt,
                                            beta=args.beta, repetition_penalty=args.repetition_penalty,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size)
                out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]

                for i in out_list:
                    with open('cnndm/att_b_{}.txt'.format(args.beta), 'a', encoding='utf8') as fw:
                        fw.write(i)
                        fw.write('\n')
            else:
                summary_ids = bart.generate(inp, attention_mask=inp_mask, num_beams=args.beam_size,
                                            max_length=args.max_length, early_stopping=True, opt_att_dist=None,
                                            beta=args.beta, repetition_penalty=args.repetition_penalty,
                                            no_repeat_ngram_size=args.no_repeat_ngram_size)
                out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summary_ids]

                for i in out_list:
                    with open('cnndm/vanilla.txt', 'a', encoding='utf8') as fw:
                        fw.write(i)
                        fw.write('\n')

            # for j in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7, 0.8, 0.9, 1, 0]:
            #     summary_ids = bart.generate(inp, attention_mask=inp_mask, num_beams=4, max_length=200, early_stopping=True, opt_att_dist=opt, beta=j)
            #
            #     out_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            #
            #     for i in out_list:
            #         with open('cnndm/abs_b{}.txt'.format(j), 'a', encoding='utf8') as fw:
            #             fw.write(i)
            #             fw.write('\n')

            # print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])