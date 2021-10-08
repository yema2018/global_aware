# Global-aware Inference
Global-aware Beam Search for Neural Abstractive Summarization (NeurIPS/NIPS 2021)

## Download the modified transformers package
As mentioned in the paper, the code of global-aware inference is based on the generation part of [HuggingFace transformers](https://github.com/huggingface/transformers/blob/v3.3.1/src/transformers/generation_utils.py), 
you can download our modified transformers from [here](https://drive.google.com/file/d/1ssonK3onfMF2Zs2gUApNz6D_NlHDMY-9/view?usp=sharing), then install the package by pip, i.e.,
```
pip install -e PATH/TO/transformers-global-aware
```   
or install by
```
python PATH/TO/transformers-global-aware/setup.py install
```
## Download processed data and checkpoints
Click [here](https://drive.google.com/file/d/1x0X2R9_I3qvb1LkeXzAxprvHXVDlK-_4/view?usp=sharing) to download processed datasets (CNN/DM, XSUM, NewsRoom, BillSum, arXiv, PubMed, Reddit_Tifu, Multi-News, WikiHow) and the checkpoint of each attention-prediction model.  

## Generation ([complete outputs](https://drive.google.com/file/d/1g9xD0jOBNqiI08TD1cBcbra4AsYnfHRU/view?usp=sharing) are provided.)
Generate with global-aware (BART): 
```
python inference.py --global_aware --beta 12 --gamma 1 --dataset cnndm --ckpt cnndm/bart_+ --cuda 0 --beam_size 4
```  
Generate with global-aware (PEGASUS): \
Just adding '--peg', you can change the summarization model from BART to PEGASUS. Same below.
```
python inference.py --global_aware --peg --beta 12 --gamma 1 --dataset newsroom --ckpt newsroom/peg_+ --cuda 0 --beam_size 8
```  
Generate with beam search. Use default config (i.e. optimal hyper-parameters of beam search) provided by BART or PEGASUS: 
```
python inference.py --vanilla --dataset cnndm --cuda 0 --beam_size 4
``` 
Generate with ORACLE global-aware:
```
python inference.py --oracle --beta 100 --gamma 1 --dataset cnndm --cuda 0 --beam_size 4
```  
The output is a text file as follows:
```
Voters in the UK go to the polls to elect a new prime minister . The Queen is officially head of state, but she's only nominally in charge . The election could result in the handing of power to the Labour Party .
Julian Zelizer: With Congress gridlocked, it's time for liberals to look local . Zelizer: At the state and local level, liberals have found more political space to move forward . He says New York City has launched an ambitious pre-K education program . Zelizer: The drive for same-sex marriage equality took hold in the states .
```

## Evaluation
Similar to BART, [files2rouge](https://github.com/pltrdy/files2rouge) is used as the evaluation matrix. The results of CNN/DMM are as follows:
```
global_beta12.0_beam4_ga1.5 (global-aware)
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
python inference.py --train --dataset cnndm --ckpt cnndm/bart_new --batch_size 36 --epoch 5
```
or
```
python inference.py --train --peg --dataset newsroom --ckpt newsroom/peg_new --batch_size 36 --epoch 5
```

