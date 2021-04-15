from att_pred_model import PreAttModel
from adabelief_pytorch import AdaBelief
import torch
import torch.nn as nn
import time
from generate_batch import gen_bt
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration


class TrainPreAtt(object):
    def __init__(self, summ_model, tokenizer, ckpt, epoch, bs, dataset, use_peg):
        self.epoch = epoch
        self.tokenizer = tokenizer
        self.bs = bs
        self.ckpt = ckpt
        self.dataset = dataset
        self.peg = use_peg
        if use_peg and dataset in ['xsum','newsroom']:
            self.ml = 512
        else:
            self.ml = 1024

        self.parallel_loss = ParallelLoss(summ_model, self.ckpt)
        self.pre_att_model = self.parallel_loss.pre_att_model
        self.optimizer = self.parallel_loss.optimizer
        # self.sch = torch.optim.lr_scheduler.StepLR(self.optimizer, 5000, gamma=0.9, last_epoch=-1)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.parallel_loss = torch.nn.DataParallel(self.parallel_loss)

        # lr = get_cosine_schedule_with_warmup(torch.optim.Adam, 50000, 100000)
        # self.optimizer = torch.optim.Adam(self.pre_att_model.parameters())

        self.device = torch.device('cuda')
        # self.pos = positional_encoding(2000, 1024).to(self.device)
        self.parallel_loss.to(self.device)

    def train(self):
        for epoch in range(self.epoch):
            total_loss = []
            start_time = time.time()
            self.pre_att_model.train()
            print('start training')
            batch_set = gen_bt(self.bs, self.tokenizer, 'train', shuffle=True, dataset=self.dataset, ml=self.ml, peg=self.peg)
            for (batch, batch_contents) in enumerate(batch_set):
                inp, tar, inp_mask, tar_mask = batch_contents
                inp = inp.to(self.device)
                tar = tar.to(self.device)
                inp_mask = inp_mask.to(self.device)
                tar_mask = tar_mask.to(self.device)
                # pos = torch.repeat_interleave(self.pos, int(inp.shape[0]), dim=0)

                self.optimizer.zero_grad()

                loss = self.parallel_loss(inp, tar, inp_mask, tar_mask).mean()
                loss.backward()
                self.optimizer.step()
                # self.sch.step()

                total_loss.append(loss.item())
                if batch % 50 == 0 and batch > 0:
                    elapsed = time.time() - start_time
                    cur_loss = np.mean(total_loss)
                    print('| epoch {:3d} | {:5d} batches | '
                          ' ms/batch {:5.2f} | '
                          'loss {:5.8f} | ppl {:8.8f}'.format(
                        epoch, batch,
                        elapsed * 1000 / len(total_loss), cur_loss, np.exp(cur_loss)))
            cur_loss = np.mean(total_loss)
            torch.save(self.pre_att_model.state_dict(), '{}_{}'.format(self.ckpt, epoch))
            elapsed = time.time() - start_time
            print('| epoch {:3d} | '
                  ' ms/epoch {:5.2f} | '
                  'loss {:5.8f} | ppl {:8.8f}'.format(
                epoch, elapsed * 1000, cur_loss, np.exp(cur_loss)))

            print('\nstart validation')
            val_loss = []
            self.pre_att_model.eval()
            val_batch = gen_bt(self.bs, self.tokenizer, mode='val', dataset=self.dataset, shuffle=True, ml=self.ml, peg=self.peg)
            for (b, bc) in enumerate(val_batch):
                inp, tar, inp_mask, tar_mask = bc
                inp = inp.to(self.device)
                tar = tar.to(self.device)
                inp_mask = inp_mask.to(self.device)
                tar_mask = tar_mask.to(self.device)
                # pos = torch.repeat_interleave(self.pos, int(inp.shape[0]), dim=0)
                with torch.no_grad():
                    loss = self.parallel_loss(inp, tar, inp_mask, tar_mask, False).mean()
                    val_loss.append(loss.item())

            print('Validation Loss {:.8f}'.format(np.mean(val_loss)))
            with open('validation for {}.txt'.format(self.dataset), 'a') as fw:
                fw.write('{}_{}: {}'.format(self.ckpt, epoch, np.mean(val_loss)))
                fw.write('\n')


class ParallelLoss(nn.Module):
    def __init__(self, summ_model, ckpt):
        super(ParallelLoss, self).__init__()
        self.loss_mse = torch.nn.MSELoss(reduction='mean')
        self.loss = EucDistanceLoss(1)
        self.summ = summ_model
        self.summ.eval()

        self.pre_att_model = PreAttModel(layers=2, d_model=1024, num_heads=16, dff=4096, rate=0.0)
        lr = 2e-5

        try:
            self.pre_att_model.load_state_dict(torch.load(ckpt))
            print('load {}'.format(ckpt))
        except:
            print('no checkpoints now!')

        self.optimizer = AdaBelief(self.pre_att_model.parameters(), lr=lr, eps=1e-16, betas=(0.9, 0.999),
                                   weight_decay=1e-4, weight_decouple=True, rectify=True)
        # self.optimizer = torch.optim.SGD(self.pre_att_model.parameters(), lr=1e-5, weight_decay=1e-4, momentum=0.9)

    def forward(self, inp, tar, inp_mask, tar_mask, training=True):
        with torch.no_grad():
            summ_output = self.summ(input_ids=inp, attention_mask=inp_mask, decoder_input_ids=tar[:, :-1],
                                    output_attentions=True,
                                    output_hidden_states=True, return_dict=True)

            decoder_att = summ_output.decoder_attentions
            opt_att_dist = 0
            tar_mask1 = tar_mask[:, 1:].unsqueeze(1).unsqueeze(-1)
            for _ in decoder_att:
                opt_att_dist += _[:, :, :, -inp.size()[1]:]
            # opt_att_dist = decoder_att[0][:, :, :, -inp.size()[1]:]
            opt_att_dist *= tar_mask1
            opt_att_dist = torch.sum(torch.mean(opt_att_dist, dim=1), dim=1)
            opt_att_dist /= len(decoder_att)

            last_encoder_hidden = summ_output.encoder_last_hidden_state

        pre_att_dist = self.pre_att_model(last_encoder_hidden, inp_mask)
        # print(torch.sum(pre_att_dist[:,:]))
        # print(torch.sum(opt_att_dist[:,:]))

        loss = self.loss(pre_att_dist, opt_att_dist)

        return loss


class EucDistanceLoss(nn.Module):
    def __init__(self, mom):
        super(EucDistanceLoss, self).__init__()
        self.mom = mom

    def forward(self, pred, target):
        return torch.mean(torch.sqrt(torch.sum((pred-target)**2, dim=-1)))


class LargePreAtt(nn.Module):
    def __init__(self):
        super(LargePreAtt, self).__init__()
        path = '/root/yema/bart-large'
        self.bart_ec = BartForConditionalGeneration.from_pretrained(path, use_cache=False).get_encoder()
        self.out_layer = nn.Linear(1024, 1)

    def forward(self, inp, mask):
        h = self.bart_ec(inp, mask, return_dict=True).last_hidden_state
        h = self.out_layer(h)
        mask = self.create_padding_mask(mask)
        logits = h.squeeze(-1)
        # mask = mask.view(batch, -1)
        logits += mask * -1e19
        att_ratio = logits.softmax(1)

        return att_ratio

    def create_padding_mask(self, ori_mask):
        mask = torch.eq(ori_mask, 0).type(torch.int)
        return mask  # (batch_size, seq_len)


if __name__ == '__main__':
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn', use_cache=False)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    a = TrainPreAtt(bart, tokenizer, 'cnndm/layer2+_+', 1, 1, 'cnndm')
    a.train()

