import torch
import torch.nn as nn
import torch.nn.functional as F
from rnnt.transformer import DecoderLayer
from rnnt.transformer import RelLearnableDecoderLayer
from rnnt.transformer import MyDecoderLayer


class BaseEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, dropout, mem_len=4, bidirectional=True):
        super(BaseEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                DecoderLayer(n_head, d_model, d_head, d_inner, dropout))

        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.drop = nn.Dropout(dropout)

        #self.mem_len = mem_len
        #self.forward_layer = nn.Linear(self.mem_len * d_model, d_model)

        # lstm
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_inner,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout if n_layer > 1 else 0,
            bidirectional=bidirectional
        )
        self.output_proj = nn.Linear(2 * d_inner if bidirectional else d_inner, d_model, bias=True)

    def forward(self, inputs, input_lengths, enc_attn_mask=None, mems=None):

        assert inputs.dim() == 3
        '''
        outputs = torch.ones(inputs.size(0), inputs.size(1), self.d_model).cuda()
        max_length = torch.max(input_lengths)

        for i in range(inputs.size(0)):
            raw_inputs = inputs[i][0:input_lengths[i]]
            core_out = raw_inputs
            for x, layer in enumerate(self.layers):
                core_out = layer(core_out, enc_attn_mask)

            outputs[i] = F.pad(core_out, pad=[0,0,0,(max_length - input_lengths[i])], value=0).cuda()

        

#        hids = []
#        core_out = self.drop(inputs)
#        hids.append(core_out)
        # rel parameters
        #r_emb = nn.Parameter(torch.ones(self.n_layer, inputs.size(1), self.n_head, self.d_head)).cuda()
        #r_w_bias = nn.Parameter(torch.ones(self.n_layer, self.n_head, self.d_head)).cuda()
        #r_bias = nn.Parameter(torch.ones(self.n_layer, inputs.size(1), self.n_head)).cuda()
        # begin to train deep nn
        
        
        core_out = inputs.permute(1, 0, 2).cuda()

        for i, layer in enumerate(self.layers):

            core_out = layer(core_out, enc_attn_mask)

        outputs = core_out.permute(1, 0, 2)
        
        outputs = torch.ones(inputs.size(1), inputs.size(0), inputs.size(2)).cuda()
        mem_len = self.mem_len
        qlen = inputs.size(1)
        
        his = 0
        while his <= qlen - 1:
            mlen = 1 if mems is not None else 0
            klen = mem_len + mlen

            if qlen - (his + mem_len) >= 0:
                seg_inputs = inputs[:, his: his + mem_len]
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
                seg_inputs = inputs[:, his:qlen]
                core_out = seg_inputs.permute(1, 0, 2)
                mem_len = seg_inputs.size(1)
                klen = mem_len + mlen

                attn_mask = torch.triu(inputs.new_ones(mem_len, klen), diagonal=1 + mlen).cuda().bool()[:, :, None]
                for i, layer in enumerate(self.layers):
                    outputs[his: qlen] = layer(core_out, dec_attn_mask=attn_mask, mems=mems)
                his = qlen

        outputs = outputs.permute(1, 0, 2)
        '''
        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        outputs = self.output_proj(outputs)

        return outputs, hidden


#class BuildEncoder(nn.Module):
#    def __init__(self, config):
#        super(BuildEncoder, self).__init__()

#        self.layers = nn.ModuleList([BaseEncoder(
#            k_len=config.enc.d_model,
#            n_layer=config.enc.n_layer,
#            n_head=config.enc.n_head,
#            d_model=config.enc.d_model,
#            d_head=config.enc.d_head,
#            d_inner=config.enc.d_inner,
#            dropout=config.dropout)
#            for i in range(config.enc.n_layer)])

#    def forward(self, inputs, input_lengths):
#        for layer in self.layers:
#            x = layer(inputs, input_lengths)

#        return x


def build_encoder(config):
    if config.enc.type == 'attention':
        return BaseEncoder(
            n_layer=config.enc.n_layer,
            n_head=config.enc.n_head,
            d_model=config.enc.d_model,
            d_head=config.enc.d_head,
            d_inner=config.enc.d_inner,
            dropout=config.dropout
        )
    else:
        raise NotImplementedError
