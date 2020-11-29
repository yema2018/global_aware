from att_pred_model import PreAttModel
from adabelief_pytorch import AdaBelief
import torch
import torch.nn as nn
import time
from generate_batch import gen_bt
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration, get_cosine_schedule_with_warmup


class TrainPreAtt(object):
    def __init__(self, summ_model, tokenizer, ckpt, epoch, bs, dataset, large, trunc):
        self.epoch = epoch
        self.tokenizer = tokenizer
        self.bs = bs
        self.ckpt = '{}/{}'.format(dataset, ckpt)
        self.dataset = dataset

        self.parallel_loss = ParallelLoss(summ_model, self.ckpt, dataset, large, trunc)
        self.pre_att_model = self.parallel_loss.pre_att_model
        self.optimizer = self.parallel_loss.optimizer

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
            batch_set = gen_bt(self.bs, self.tokenizer, 'train', shuffle=True, dataset=self.dataset)
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
            val_batch = gen_bt(self.bs, self.tokenizer, mode='test', dataset=self.dataset)
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

            print('Attention-aware Score {:.8f}'.format(np.mean(val_loss)))
            with open('records for {}.txt'.format(self.ckpt.replace('/', '_')), 'a') as fw:
                fw.write('epoch{}: {}'.format(epoch, np.mean(val_loss)))
                fw.write('\n')


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype=torch.float32)


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


class ParallelLoss(nn.Module):
    def __init__(self, summ_model, ckpt, dataset, large, trunc):
        super(ParallelLoss, self).__init__()
        self.loss_mse = torch.nn.MSELoss(reduction='mean')
        self.loss = EucDistanceLoss(1)
        self.summ = summ_model
        self.summ.eval()
        self.large = large
        self.trunc = trunc
        if large:
            self.pre_att_model = LargePreAtt(dataset)
            lr = 2e-5
        else:
            self.pre_att_model = PreAttModel(layers=2, d_model=1024, num_heads=16, dff=4096, rate=0.0)
            lr = 1e-4

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
                                    decoder_attention_mask=tar_mask[:, :-1], output_attentions=True,
                                    output_hidden_states=True)

            decoder_att = summ_output[2]
            opt_att_dist = 0
            tar_mask1 = tar_mask[:, 1:].unsqueeze(1).unsqueeze(-1)
            for _ in decoder_att:
                opt_att_dist += _[:, :, :, -inp.size()[1]:]
            opt_att_dist *= tar_mask1
            opt_att_dist = torch.mean(torch.mean(opt_att_dist, dim=1), dim=1)
            opt_att_dist = opt_att_dist[:, :self.trunc]
            opt_att_dist /= torch.sum(opt_att_dist, dim=1, keepdim=True)

            last_encoder_hidden = summ_output[3]
        if self.large:
            pre_att_dist = self.pre_att_model(inp[:, :self.trunc], inp_mask[:, :self.trunc])
        else:
            pre_att_dist = self.pre_att_model(last_encoder_hidden[:, :self.trunc, :], inp_mask[:, :self.trunc])
        # print(pre_att_dist[:, :20])
        # print(opt_att_dist[:, :20])
        if training:
            # loss = 0
            # for i in range(int(last_encoder_hidden.shape[0])):
            #     loss += -torch.log(cos_sim(pre_att_dist[i], opt_att_dist[i]))
            # loss /= last_encoder_hidden.shape[0]
            loss = self.loss(pre_att_dist, opt_att_dist)
            # loss = -torch.mean(torch.log(torch.sum(torch.min(opt_att_dist, pre_att_dist), dim=-1)))
            # loss = self.loss_mse(pre_att_dist.view(-1), opt_att_dist.view(-1))
        else:
            loss = torch.mean(torch.log(torch.sum(torch.min(opt_att_dist, pre_att_dist), dim=-1)))

        return loss


class EucDistanceLoss(nn.Module):
    def __init__(self, mom):
        super(EucDistanceLoss, self).__init__()
        self.mom = mom

    def forward(self, pred, target):
        return torch.mean(torch.sum((pred-target)**2, dim=-1))


class LargePreAtt(nn.Module):
    def __init__(self, dataset):
        super(LargePreAtt, self).__init__()
        # if dataset == 'cnndm':
        #     path = '/root/yema/bart-large-cnn'
        # elif dataset == 'xsum':
        #     path = '/root/yema/bart-large-xsum'
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

def cos_sim(a, b):
    a = a.view(-1)
    b = b.view(-1)
    dot = torch.dot(a, b)
    a_ = torch.sqrt(torch.sum(a**2))
    b_ = torch.sqrt(torch.sum(b ** 2))

    return (dot/(a_ * b_))/2+0.5


if __name__ == '__main__':
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum', use_cache=False)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
    a = TrainPreAtt(bart, tokenizer, 'layer2_eud_9_1', 1, 1, 'xsum', False, 1024)
    a.train()

