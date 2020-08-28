import torch
import torch.nn as nn
import torch.nn.functional as F
from rnnt.encoder import build_encoder
from rnnt.decoder import build_decoder
from warprnnt_pytorch import RNNTLoss
from warpctc_pytorch import CTCLoss
import math


class Pre_encoder(nn.Module):
    def __init__(self, config):
        super(Pre_encoder, self).__init__()
        # define encoder
        self.config = config

        # self.encoder = BuildEncoder(config)
        self.encoder = build_encoder(config)
        self.project_layer = nn.Linear(800, 2664)

        self.crit = CTCLoss()

    def forward(self, inputs, inputs_length, targets, targets_length, mems=None):

        logits = self.encoder(inputs, inputs_length)
        logits = self.project_layer(logits)
        #logits = F.softmax(logits, dim=2)
        logits = logits.cpu()
        targets = targets.cpu()
        inputs_length = inputs_length.cpu()
        targets_length = targets_length.cpu()

        loss = self.crit(logits.permute(1, 0, 2), targets.int().view(-1), inputs_length.int(), targets_length.int())

        return loss


class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:

            enc_state = enc_state.unsqueeze(2)
            dec_state = dec_state.unsqueeze(1)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs


class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
        # define encoder
        self.config = config

        # self.encoder = BuildEncoder(config)
        self.encoder = build_encoder(config)
        self.project_layer = nn.Linear(320, 213)
        # define decoder
        self.decoder = build_decoder(config)
        # define JointNet
        self.joint = JointNet(
            input_size=config.joint.input_size,
            inner_dim=config.joint.inner_size,
            vocab_size=config.vocab_size
            )

        if config.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), '%d != %d' % (self.decoder.embedding.weight.size(1),  self.joint.project_layer.weight.size(1))
            self.joint.project_layer.weight = self.decoder.embedding.weight
        #self.ctc_crit = CTCLoss()
        self.rnnt_crit = RNNTLoss()

    def forward(self, inputs, inputs_length, targets, targets_length, mems=None):

        enc_state, _ = self.encoder(inputs, inputs_length)
#       ctc_logits = self.project_layer(enc_state)
        #logits = F.softmax(logits, dim=2)
#        ctc_logits = ctc_logits.cpu()
#        targets = targets.cpu()
#        inputs_length = inputs_length.cpu()
#        targets_length = targets_length.cpu()

#        ctc_loss = self.ctc_crit(ctc_logits.permute(1, 0, 2), targets.int().view(-1), inputs_length.int(), targets_length.int())

        concat_targets = F.pad(targets.cuda(), pad=[1, 0, 0, 0], value=0)
        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))

        logits = self.joint(enc_state, dec_state)
        #logits = F.softmax(logits, dim=3)
#        logits = logits.cpu()
#        targets = targets.cpu()
        #inputs_length = inputs_length.cuda()
        #targets_length = targets_length.cuda()

        loss = self.rnnt_crit(logits, targets.int(), inputs_length.int(), targets_length.int())

#
        #loss = 0.3 * ctc_loss + 0.7 * rnnt_loss

        return loss

    def recognize(self, inputs, inputs_length):

        batch_size = inputs.size(0)

        enc_states, _ = self.encoder(inputs, inputs_length)

        zero_token = torch.LongTensor([[0]])
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []
            dec_state, hidden = self.decoder(zero_token)
#            B = {'0': 1}

            for t in range(lengths):
#                A = B
#                B = {}

                # add prefix probability
#                for y in A:
#                    for x in A:
#                        if (x != y) & y.startswith(x):
#                            pre = x.split(' ')
#                            dif = list(set(y.split(' '))-set(pre))

#                            prob_pre = 1
#                            for i in range(len(dif)):
#                                pre_token = torch.LongTensor([[int(x) for x in pre]]).cuda()
#                                pre_token = F.pad(pre_token, pad=[1, 0, 0, 0], value=0)
#                                dec_state = self.decoder(pre_token)

#                                dec_state = dec_state.squeeze(0)
#                                dec_state = dec_state[dec_state.size(0) - 1]

#                                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
#                                out = F.softmax(logits, dim=0).detach()

#                                prob_pre *= out[int(dif[i])].item()
#                                pre.append(str(dif[i]))

#                            A[y] += A[x] * prob_pre

#                flag = True
#                while flag:
#                    count = 1
#                    for w in B:
#                        if B[w] > A[max(A)]:
#                            count += 1

#                    if count > 2:
#                        break

#                    ystar = max(A, key=A.get)
#                    prob = A[ystar]
#                    del A[ystar]

#                    ystar_list = ystar.split(' ')
#                    ystar_token = torch.LongTensor([[int(x) for x in ystar_list]]).cuda()

#                    dec_state = self.decoder(ystar_token)
#                    dec_state = dec_state.squeeze(0)
#                    dec_state = dec_state[dec_state.size(0) - 1]

#                    logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
#                    out = F.softmax(logits, dim=0).detach()

                    # null label
#                    B[ystar] = prob * (out[0].item())

                    # k labels
#                    k = 1
#                    while k < 1040:
#                        ystar_list.append(str(k))
#                        A[" ".join(ystar_list)] = prob * (out[k].item())
#                        ystar_list.pop()
#                        k += 1

                # remove all but the W most probable from B
#                v2k = {}
#                for key in B:
#                    v2k[B[key]] = key

#                sorted_values = [v for v in sorted(B.values(), reverse=True)]
#                B = {}

#                i = 0
#                for unit in sorted_values:
#                    if i < 2:
#                        B[v2k[unit]] = unit
#                        i += 1

#            for unit in B:
#                B[unit] = math.log(B[unit] / len([unit]))
#            token_list = max(B, key=B.get).split(' ')
#            del token_list[0]

                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0)
                pred = int(pred.item())

                if pred != 0:
                    token_list.append(pred)

                    token = torch.LongTensor([[pred]]).cuda()
                    #token = torch.LongTensor([token_list]).cuda()
                    #token = F.pad(token, pad=[1, 0, 0, 0], value=0)

                    if enc_state.is_cuda:
                        token = token.cuda()

                    dec_state, hidden = self.decoder(token, hidden=hidden)
                    #dec_state = dec_state.squeeze()
                    #dec_state = dec_state[dec_state.size(0) - 1]

                    #logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                    #out = F.softmax(logits, dim=0).detach()
                    #pred = torch.argmax(out, dim=0)
                    #pred = int(pred.item())

            return token_list
        # single seq for recognize
        results = []
        for i in range(batch_size):
            decoded_seq = decode(enc_states[i], inputs_length[i])
            results.append(decoded_seq)

        with open('decode.txt', 'a') as fid:
            for line in results:
                fid.write(str(line)+'\n')

        return results
