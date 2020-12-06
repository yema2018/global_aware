import json
import numpy as np
import torch


def gen_bt(bs, tokenizer, mode, dataset='cnndm', shuffle=False):
    data = [json.loads(i) for i in open('{}/{}.json'.format(dataset, mode), encoding='utf8')]
    tgt = [i['tgt'] for i in data]

    if dataset == 'cnndm':
        src = []
        for i in data:
            text = i['src']
            if text[:5] == '(CNN)':
                text = text[5:]
            src.append(text)
    else:
        src = [i['src'] for i in data]

    if shuffle:
        cc = list(zip(src, tgt))
        np.random.shuffle(cc)
        src[:], tgt[:] = zip(*cc)

    length = len(data)
    num_batch = int(np.ceil(length / bs))

    if dataset == 'wmt':
        for i in range(num_batch):
            begin = i * bs
            stop = min((i + 1) * bs, length)
            source = src[begin:stop]
            target = tgt[begin:stop]

            batch = tokenizer.prepare_seq2seq_batch(source, src_lang="en_XX", tgt_lang="ro_RO",
                                                    tgt_texts=target, max_length=64, max_target_length=64, padding=True,
                                                    truncation=True)
            input_ids = batch["input_ids"]
            target_ids = batch["labels"]
            inp_mask = batch['attention_mask']
            tar_mask = 1 - torch.eq(target_ids, 1).type(torch.int)
            yield input_ids, target_ids, inp_mask, tar_mask
    else:
        for i in range(num_batch):
            begin = i * bs
            stop = min((i+1)*bs, length)
            source = src[begin:stop]
            target = tgt[begin:stop]

            sources = tokenizer(source, return_tensors='pt', max_length=1024, padding=True, truncation=True)
            targets = tokenizer(target, return_tensors='pt', max_length=150, padding=True, truncation=True)

            yield sources['input_ids'], targets['input_ids'], sources['attention_mask'], targets['attention_mask']


if __name__ == '__main__':
    from transformers import MBartForConditionalGeneration, MBartTokenizer

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro", use_cache=False)
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro")
    gen_bt(4, tokenizer, 'val', dataset='wmt', shuffle=False)