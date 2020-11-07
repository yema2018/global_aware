from att_prediction import PreAtt
from transformers import BartForConditionalGeneration, BartTokenizer
from generate_batch import gen_bt
import torch

bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn', use_cache=False)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
a = PreAtt(bart, tokenizer, './cnndm/layer_1_0', 10, 3)

att_pre_model = a.pre_model()
print(att_pre_model)

device = torch.device('cuda:0')
batch_set = gen_bt(1, tokenizer, 'test')
for (batch, batch_contents) in enumerate(batch_set):
    inp, _, inp_mask, _ = batch_contents
    inp = inp.to(device)
    inp_mask = inp_mask.to(device)
    # print(inp)
    summary_ids = bart.generate(inp, attention_mask=inp_mask, num_beams=4, max_length=200, early_stopping=True, att_pre_model=att_pre_model, beta=0.8)
    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])