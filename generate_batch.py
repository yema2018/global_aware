import json
import numpy as np
import torch


def gen_bt(bs, tokenizer, mode, dataset='cnndm', shuffle=False, ml=1024, peg=False):
    data = [json.loads(i) for i in open('{}/{}.json'.format(dataset, mode), encoding='utf8')]
    tgt = [i['tgt'] for i in data]

    if dataset == 'cnndm':
        src = []
        for i in data:
            text = i['src']
            if text[:5] == '(CNN)':
                text = text[5:]
                if text[:4] == ' -- ':
                    text = text[4:]
            src.append(text)
    else:
        src = [i['src'] for i in data]

    if shuffle:
        cc = list(zip(src, tgt))
        np.random.shuffle(cc)
        src[:], tgt[:] = zip(*cc)

    length = len(data)
    num_batch = int(np.ceil(length / bs))

    for i in range(num_batch):
        begin = i * bs
        stop = min((i+1)*bs, length)
        source = src[begin:stop]
        target = tgt[begin:stop]

        sources = tokenizer(source, return_tensors='pt', max_length=ml, padding=True, truncation=True)
        targets = tokenizer(target, return_tensors='pt', max_length=256, padding=True, truncation=True)
        tar_ids = targets['input_ids']
        tar_mask = targets['attention_mask']
        src_ids = sources['input_ids']
        src_mask = sources['attention_mask']

        if peg:
            prefix = torch.tensor([0]).unsqueeze(0).repeat_interleave(tar_ids.shape[0], 0)
            tar_ids = torch.cat((prefix, tar_ids), 1)
            prefix = torch.tensor([1]).unsqueeze(0).repeat_interleave(tar_mask.shape[0], 0)
            tar_mask = torch.cat((prefix, tar_mask), 1)

        yield src_ids, tar_ids, src_mask, tar_mask


if __name__ == '__main__':
    from transformers import MBartForConditionalGeneration, MBartTokenizer

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro", use_cache=False)
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro")
    gen_bt(4, tokenizer, 'val', dataset='wmt', shuffle=False)