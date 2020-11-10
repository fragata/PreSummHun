# PreSummHun

bert-large

train
CUDA_VISIBLE_DEVICES=4,5,6,7 python3.6 src/train.py -task abs -mode train -large -bert_data_path bert_data/cnndm -dec_dropout 0.2 -model_path models/large -sep_optim true -lr_bert 0.001 -lr_dec 0.1 -save_checkpoint_steps 20000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -max_length 1000 -visible_gpus 4,5,6,7 -log_file logs/train_abs_large

test
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 src/train.py -task abs -mode test -large -test_from models/large/model_step_200000.pt -batch_size 140 -test_batch_size 140 -bert_data_path bert_data/cnndm -log_file logs/test_abs_large -model_path models/large -sep_optim false -use_interval true -visible_gpus 0,1,2,3 -max_pos 512 -alpha 0.95 -result_path results/large

pi.itk.ppke.hu/demo/summarize/data/bert_data.tar.gz
pi.itk.ppke.hu/demo/summarize/data/bert-large.tar.gz

Mappastruktúra:
.
├── bert_data
│   ├── cnndm.test.0.bert.pt
│   ├── cnndm.train.0.bert.pt
│   ├── cnndm.train.1.bert.pt
│   ├── cnndm.train.2.bert.pt
│   ├── cnndm.train.3.bert.pt
│   └── cnndm.train.4.bert.pt
├── bert-large
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── bert-large.tar.gz
├── logs
│   └── abs-large
├── models
├── README.md
├── requirements.txt
├── results
├── src