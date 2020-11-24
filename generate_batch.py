import json
import numpy as np


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
    for i in range(num_batch):
        begin = i * bs
        stop = min((i+1)*bs, length)
        source = src[begin:stop]
        target = tgt[begin:stop]

        sources = tokenizer(source, return_tensors='pt', max_length=1024, padding=True, truncation=True)
        targets = tokenizer(target, return_tensors='pt', max_length=150, padding=True, truncation=True)

        yield sources['input_ids'], targets['input_ids'], sources['attention_mask'], targets['attention_mask']
