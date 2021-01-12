# attention-aware
Attention-aware inference: generate texts according to attention distribution.

## Download the modified transformers package
As mentioned in the paper, the code of attention-aware inference is based on the generation part of [HuggingFace transformers](https://github.com/huggingface/transformers/blob/v3.3.1/src/transformers/generation_utils.py), 
you can download our modified transformers from [here](https://drive.google.com/file/d/19dYBwwcXTTRmWwBBeWbGZtcQ5jHTG_g1/view?usp=sharing), then install the package by pip, i.e.,
```
pip install -e PATH/TO/transformers-att-aware
```    
## Download processed data and checkpoints
Click [here](https://drive.google.com/file/d/1s4dMpZp3EgkinzEIuZoVwk2A7d3NP4HA/view?usp=sharing) to download processed datasets (CNN/DM, XSUM, WMT'16, NEWSRoom) and the checkpoint of each attention-prediction model.  

## Generation (complete outputs are provided.)
Generate with attention-aware: 
```
python bart_gen.py --att_aware --beta 12 --gamma 1.5 --dataset cnndm --ckpt cnndm/layer2+_+ --cuda 0 --beam_size 4
```  
Generate with vanilla beam search: 
```
python bart_gen.py --vanilla --dataset cnndm --cuda 0 --beam_size 4
``` 
Generate with cheating attention-aware:
```
python bart_gen.py --cheat --beta 100 --gamma 1 --dataset cnndm --cuda 0 --beam_size 4
```  
The output is a text file as follows:
```
Voters in the UK go to the polls to elect a new prime minister . The Queen is officially head of state, but she's only nominally in charge . The election could result in the handing of power to the Labour Party .
Julian Zelizer: With Congress gridlocked, it's time for liberals to look local . Zelizer: At the state and local level, liberals have found more political space to move forward . He says New York City has launched an ambitious pre-K education program . Zelizer: The drive for same-sex marriage equality took hold in the states .
```
You can find the complete text files of att-aware and beam search of each dataset from [here](https://drive.google.com/file/d/1GIsYYJcaUA_PVsMGL9C1ue1vqlFJwSuu/view?usp=sharing)  

## Evaluation
Similar to BART, [files2rouge](https://github.com/pltrdy/files2rouge) is used as the evaluation matrix. The results of CNN/DMM are as follows:
```
att_beta12.0_beam4_ga1.5 (att-aware)
---------------------------------------------
1 ROUGE-1 Average_R: 0.45689 (95%-conf.int. 0.45445 - 0.45928)
1 ROUGE-1 Average_P: 0.46657 (95%-conf.int. 0.46389 - 0.46904)
1 ROUGE-1 Average_F: 0.45134 (95%-conf.int. 0.44921 - 0.45347)
---------------------------------------------
1 ROUGE-2 Average_R: 0.22000 (95%-conf.int. 0.21758 - 0.22252)
1 ROUGE-2 Average_P: 0.22576 (95%-conf.int. 0.22329 - 0.22852)
1 ROUGE-2 Average_F: 0.21768 (95%-conf.int. 0.21535 - 0.22019)
---------------------------------------------
1 ROUGE-L Average_R: 0.42546 (95%-conf.int. 0.42298 - 0.42781)
1 ROUGE-L Average_P: 0.43457 (95%-conf.int. 0.43200 - 0.43721)
1 ROUGE-L Average_F: 0.42035 (95%-conf.int. 0.41817 - 0.42260)
```
```
vanilla_beam4 (beam search)
---------------------------------------------
1 ROUGE-1 Average_R: 0.51491 (95%-conf.int. 0.51210 - 0.51760)
1 ROUGE-1 Average_P: 0.40264 (95%-conf.int. 0.40019 - 0.40500)
1 ROUGE-1 Average_F: 0.44119 (95%-conf.int. 0.43902 - 0.44325)
---------------------------------------------
1 ROUGE-2 Average_R: 0.24733 (95%-conf.int. 0.24449 - 0.25028)
1 ROUGE-2 Average_P: 0.19389 (95%-conf.int. 0.19162 - 0.19631)
1 ROUGE-2 Average_F: 0.21208 (95%-conf.int. 0.20964 - 0.21452)
---------------------------------------------
1 ROUGE-L Average_R: 0.47704 (95%-conf.int. 0.47425 - 0.47974)
1 ROUGE-L Average_P: 0.37331 (95%-conf.int. 0.37094 - 0.37572)
1 ROUGE-L Average_F: 0.40893 (95%-conf.int. 0.40677 - 0.41100)
```  

## Train attention-prediction model
Someone who want to re-train the attention-prediction model can execute the following command.
```
python bart_gen.py --train --dataset cnndm --ckpt cnndm/layer2_new --batch_size 36 --epoch 5
```

