from transformers import BartTokenizer,BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn', use_cache=False)

sen = 'I am ye ma from xjtlu.'
sen2 = 'yema qq 11'

enc = tokenizer(sen, return_tensors='pt')
dec = tokenizer(sen2, return_tensors='pt')

bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn', use_cache=False)

out = bart(**enc, output_attentions=True, output_hidden_states=True)
out1 = bart(**enc, decoder_input_ids=dec['input_ids'][:, :-1],output_attentions=True, output_hidden_states=True )

decoder_att = out[2]
last_encoder_hidden = out[3]

