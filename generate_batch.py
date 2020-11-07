import json
import numpy as np


def gen_bt(bs, tokenizer, mode, data='cnndm', shuffle=False):
    data = [json.loads(i) for i in open('{}/{}.json'.format(data, mode), encoding='utf8')]
    src = [' '.join(i['src']) for i in data]
    tgt = [' '.join(i['tgt']) for i in data]

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

        sources = tokenizer(source, return_tensors='pt', max_length=800, padding=True, truncation=True)
        targets = tokenizer(target, return_tensors='pt', max_length=200, padding=True, truncation=True)

        yield sources['input_ids'], targets['input_ids'], sources['attention_mask'], targets['attention_mask']
