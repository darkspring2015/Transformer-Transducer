import torch
import torch.nn as nn
from rnnt.transformer import DecoderLayer
from rnnt.transformer import MyDecoderLayer
from rnnt.transformer import RelPartialLearnableDecoderLayer
from rnnt.transformer import AdaptiveEmbedding
from rnnt.transformer import PositionalEmbedding
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F


class BaseDecoder(nn.Module):

    def __init__(self, vocab_size, n_layer, n_head, d_model, d_head, d_inner, dropout, mem_len=4, ext_len=0):
        super(BaseDecoder, self).__init__()

        # Transformer
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                   DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout)
            )

        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.drop = nn.Dropout(dropout)

        #self.mem_len = mem_len
        #self.ext_len = ext_len
        #self.forward_layer = nn.Linear(self.mem_len * d_model, d_model)
        # lstm
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_inner,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout if n_layer > 1 else 0
        )
        self.output_proj = nn.Linear(d_inner, d_model)

        # load pre-trained model HIT bert wwm, download from https://github.com/ymcui/Chinese-BERT-wwm
        #self.tokenizer = BertTokenizer.from_pretrained('./wwm/pytorch')
        #self.model = BertModel.from_pretrained('./wwm/pytorch')

        # load pre-trained embedding
        #self.embedding = nn.Embedding.from_pretrained(self.pretrain(), freeze=False)

        # one hot
        #self.embedding = nn.Embedding.from_pretrained(self.one_hot(vocab_size, d_model), padding_idx=0)
        #self.embedding = self.one_hot(vocab_size, d_model)

        # nn.embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # adaptive embedding
        #self.embedding = AdaptiveEmbedding(vocab_size, d_model, d_model)

#        if share_weight:
#            self.embedding.weight = self.output_proj.weight

    def one_hot(self, vocab_size, d_model):

        idx_list = [vocab_size - 1]

        for x in range(0, vocab_size - 1):
            idx_list.append(x)

        embedding = torch.eye(vocab_size, d_model)[idx_list].cuda()

        return embedding

    def pretrain(self):

        embedding = []
        with open('thchs30_train_char_embedding.txt', 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split()
                emb = [float(i) for i in parts[1:]]
                embedding.append(emb)

        embedding = torch.tensor(embedding).cuda()

        return embedding

    # Transformer-xl as follows
    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def forward(self, inputs, seq_length=None, hidden=None, mems=None):

        # one-hot
        #embed_inputs = self.embedding[inputs]

        # nn.Embedding

        embed_inputs = self.embedding(inputs)


        # adaptive embedding
        #embed_inputs = self.embedding(inputs)
        '''
        outputs = torch.ones(embed_inputs.size(0), embed_inputs.size(1), self.d_model).cuda()
        if seq_length is not None:
            max_length = torch.max(seq_length)

            for i in range(inputs.size(0)):
                raw_inputs = embed_inputs[i][0:seq_length[i]]

                for x, layer in enumerate(self.layers):
                    qlen = seq_length[i]
                    mlen = 1 if mems is not None else 0
                    klen = mlen + qlen
                    attn_mask = torch.triu(inputs.new_ones(qlen, klen), diagonal=1 + mlen).cuda().bool()[:, :, None]
                    core_out = layer(raw_inputs, attn_mask)

                outputs[i] = F.pad(core_out, pad=[0,0,0,(max_length - seq_length[i])], value=0).cuda()
        else:
            for x, layer in enumerate(self.layers):
                embed_inputs = embed_inputs.squeeze(0)
                outputs = layer(embed_inputs)
        
        outputs = torch.ones(embed_inputs.size(1), embed_inputs.size(0), embed_inputs.size(2)).cuda()
        mem_len = self.mem_len
        qlen = inputs.size(1)
        # train
        if seq_length is not None:
            his = 0
            while his <= qlen - 1:
                mlen = 1 if mems is not None else 0
                klen = mem_len + mlen

                if qlen - (his + mem_len) >= 0:
                    seg_inputs = embed_inputs[:, his: his + mem_len]
                    core_out = seg_inputs.permute(1, 0, 2)

                    attn_mask = torch.triu(inputs.new_ones(mem_len, klen), diagonal=1 + mlen).cuda().bool()[:, :, None]
                    for i, layer in enumerate(self.layers):
                        outputs[his: his + mem_len] = layer(core_out, dec_attn_mask=attn_mask, mems=mems)
                        with torch.no_grad():
                            history = outputs[his: his + mem_len]
                            A = torch.cat((history[0], history[1]), dim=1)
                            B = torch.cat((history[2], history[3]), dim=1)
                            AB = torch.cat((A,B), dim=1)
                            mems = self.forward_layer(AB)

                    his = his + mem_len
                else:
                    seg_inputs = embed_inputs[:, his:qlen]
                    core_out = seg_inputs.permute(1, 0, 2)
                    mem_len = seg_inputs.size(1)
                    klen = mem_len + mlen

                    attn_mask = torch.triu(inputs.new_ones(mem_len, klen), diagonal=1 + mlen).cuda().bool()[:, :, None]
                    for i, layer in enumerate(self.layers):
                        outputs[his: qlen] = layer(core_out, dec_attn_mask=attn_mask, mems=mems)
                    his = qlen
        # recog
        else:
            if qlen <= mem_len:
                mem_len = inputs.size(1)
                klen = mem_len
                core_out = embed_inputs.permute(1, 0, 2)
                attn_mask = torch.triu(inputs.new_ones(mem_len, klen), diagonal=1).cuda().bool()[:, :, None]
                for i, layer in enumerate(self.layers):
                    outputs = layer(core_out, dec_attn_mask=attn_mask, mems=mems)

            else:
                his = 0
                while his <= qlen - 1:
                    mlen = 1 if mems is not None else 0
                    klen = mem_len + mlen
                    if qlen - (his + mem_len) >= 0:

                        seg_inputs = embed_inputs[:, his: his + mem_len]
                        core_out = seg_inputs.permute(1, 0, 2)

                        attn_mask = torch.triu(inputs.new_ones(mem_len, klen), diagonal=1 + mlen).cuda().bool()[:, :, None]
                        for i, layer in enumerate(self.layers):
                            outputs[his: his + mem_len] = layer(core_out, dec_attn_mask=attn_mask, mems=mems)

                            with torch.no_grad():
                                history = outputs[his: his + mem_len]
                                A = torch.cat((history[0], history[1]), dim=1)
                                B = torch.cat((history[2], history[3]), dim=1)
                                AB = torch.cat((A, B), dim=1)
                                mems = self.forward_layer(AB)

                        his = his + mem_len
                    else:
                        seg_inputs = embed_inputs[:, his:qlen]
                        core_out = seg_inputs.permute(1, 0, 2)
                        mem_len = seg_inputs.size(1)
                        klen = mem_len + mlen

                        attn_mask = torch.triu(inputs.new_ones(mem_len, klen), diagonal=1 + mlen).cuda().bool()[:, :, None]
                        for i, layer in enumerate(self.layers):
                            outputs[his: qlen] = layer(core_out, dec_attn_mask=attn_mask, mems=mems)
                        his = qlen

        outputs = outputs.permute(1, 0, 2)

        # rel parameters
        #pos_seq = torch.arange(inputs.size(0) - 1, -1, -1.0).cuda()
        #pos_emb = self.pos_emb(pos_seq)
        #pos_emb = self.drop(pos_emb)
        #r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head)).cuda()
        #r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head)).cuda()
        

        # train
        if seq_length is not None:
            outputs = torch.ones(inputs.size(0), inputs.size(1), self.d_model).cuda()

            max_length = torch.max(seq_length)

            for i in range(inputs.size(0)):

                zeros = torch.zeros(max_length - seq_length[i]).cuda()
                ones = torch.ones(seq_length[i]).cuda()
                mask = torch.cat((ones, zeros), 0).bool().cuda()

                raw_inputs = inputs[i].masked_select(mask)
                #input = [unit.item() for unit in raw_inputs.flatten()]
                #input_id = torch.tensor([self.tokenizer.encode(input, add_special_tokens=False)]).cuda()

                with torch.no_grad():

                    hidden = self.model(raw_inputs.unsqueeze(0))

                outputs[i] = hidden.squeeze()
                    #outputs[i] = F.pad(hidden, pad=[0,0,1,(max_length - seq_length[i])], value=0).cuda()

        # recognize
        else:
            input = [unit.item() for unit in inputs.flatten()]
            input_id = torch.tensor([self.tokenizer.encode(input, add_special_tokens=False)]).cuda()
            with torch.no_grad():
                hidden = self.model(input_id)[0]
                outputs = hidden.squeeze()


        # recognize
        #if inputs.size(0) == 1:
        #    input = inputs.cpu().numpy().tolist()

        #        outputs = self.MultiHeadAttention(embed_targets, attn_mask)
        #    else:
        #        input_id = torch.tensor([self.tokenizer.encode(input, add_special_tokens=False)]).cuda()
        #        with torch.no_grad():
        #            hidden = self.model(input_id)[0]
        #            hidden = hidden.squeeze()

        #        attn_mask = torch.triu(hidden.new_ones(hidden.size(0), hidden.size(0)), diagonal=1).bool()[:, :, None]

        #        outputs = self.MultiHeadAttention(hidden, attn_mask)

        # train ger char embedding

        #if inputs.size(0) > 1:
        #    max_length = torch.max(seq_length)
        #else:
        #    max_length = inputs.size(0)
        #    seq_length = torch.tensor([[1]])

        #outputs = torch.ones(inputs.size(0), inputs.size(1), 768).cuda()

        #for i in range(inputs.size(0)):
        #    zeros = torch.zeros(max_length - seq_length[i]).cuda()
        #    ones = torch.ones(seq_length[i]).cuda()
        #    mask = torch.cat((zeros, ones), 0).bool().cuda()

        #    no_pad_input = inputs[i].masked_select(mask)

        #    input = [unit.item() for unit in inputs[i].flatten()]

        #    if input == [0]:
        #        embed_inputs = torch.tensor([[[0.] * 768]]).cuda()

        #    else:

        #    with torch.no_grad():
        #        hidden = self.model(input_id)[0]
        #        hidden = hidden.squeeze()
        #        hidden = F.pad(hidden, pad=[0,0,1,(max_length - seq_length[i])], value=0).cuda()

        # padding mask
        #if seq_length is not None:
        #    max_length = torch.max(seq_length)
        #    for i in range(inputs.size(0)):
        #        zeros = torch.zeros(seq_length[i]).cuda()
        #        ones = torch.ones(max_length - seq_length[i]).cuda()
        #        mask = torch.cat((zeros, ones), 0)[:, None].bool().cuda()

        #        embed_inputs[i] = embed_inputs[i].masked_fill_(mask, -float(2 ** 32))

        # begin to train deep nn (onehot, nn.embedding )
        #if not mems: mems = self.init_mems()

        # exchange the position between batch and seq_length for attention score computation
        
        core_out = embed_inputs.permute(1, 0, 2).cuda()
        #hids = []
        #hids.append(core_out)
        for i, layer in enumerate(self.layers):
            # attn mask
            qlen = core_out.size(0)
            mlen = mems[0].size(0) if mems is not None else 0
            klen = mlen + qlen
            attn_mask = torch.triu(core_out.new_ones(qlen, klen), diagonal=1+mlen).cuda().bool()[:, :, None]

            core_out = layer(core_out, dec_attn_mask=attn_mask, mems=mems)
            #hids.append(core_out)
            # make shape back

        outputs = core_out.permute(1, 0, 2)

        #new_mems = self.update_mems(hids, mems, qlen, mlen)
       '''
        # lstm
        if seq_length is not None:
            sorted_seq_lengths, indices = torch.sort(seq_length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(embed_inputs, sorted_seq_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)
        # rnn 展开
        if seq_length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        outputs = self.output_proj(outputs)

        return outputs, hidden


def build_decoder(config):
    if config.dec.type == 'attention':
        return BaseDecoder(
            vocab_size=config.vocab_size,
            n_layer=config.dec.n_layer,
            n_head=config.dec.n_head,
            d_model=config.dec.d_model,
            d_head=config.dec.d_head,
            d_inner=config.dec.d_inner,
            dropout=config.dropout
        )
    else:
        raise NotImplementedError
