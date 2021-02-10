import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MLP(nn.Module):
    def __init__(self, hidden_dim, inp_dim):
        super(MLP, self).__init__()
        self.dense_dim, self.time_dist_dim = inp_dim
        poi_embed_dim = 3
        self.poi_type_embedding = nn.Embedding(21, poi_embed_dim)
        inp_dim = poi_embed_dim + self.dense_dim + self.time_dist_dim
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, addr_type, quant, time_dist):
        x = torch.cat((self.poi_type_embedding(addr_type).squeeze(dim=1), quant, time_dist), dim=1)
        return self.net(x)


class LocMatcher(nn.Module):
    def __init__(self, hidden_dim, nb_heads, nb_layers, loc_inp_dim, dropout=0.1, use_addr=True):
        super(LocMatcher, self).__init__()
        fc_hidden_dim = 32
        self.use_addr = use_addr
        if self.use_addr:
            poi_embed_dim = 3
            self.poi_type_embedding = nn.Embedding(21, poi_embed_dim)
            addr_quant_dim = 1
            addr_inp_dim = addr_quant_dim + poi_embed_dim
            self.fc_addr = nn.Linear(addr_inp_dim, fc_hidden_dim)
        self.nb_heads = nb_heads
        self.loc_dense_dim, self.time_dist_dim = loc_inp_dim
        if self.time_dist_dim != 0:
            time_dist_embed_dim = 3
            self.time_dist_embedding = nn.Linear(self.time_dist_dim, time_dist_embed_dim)
            loc_inp_dim = self.loc_dense_dim + time_dist_embed_dim
        else:
            loc_inp_dim = self.loc_dense_dim
        self.inp_embedding = nn.Linear(loc_inp_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(hidden_dim, nb_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nb_layers)
        self.fc_output = nn.Linear(hidden_dim, fc_hidden_dim, bias=False)
        self.fc_combine = nn.Linear(fc_hidden_dim, 1, bias=False)
        self.device = torch.device('cpu')

    def forward(self, addr, addr_type, loc_dense_seq, time_dist_seq, data_length):
        nb_batches = addr.size(0)
        key_padding_mask = self.generate_key_padding_mask(data_length)
        attn_mask = self.generate_attn_mask(self.nb_heads, data_length)
        use_seq = []
        if self.loc_dense_dim != 0:
            use_seq.append(loc_dense_seq)
        if self.time_dist_dim != 0:
            use_seq.append(self.time_dist_embedding(time_dist_seq))
        loc_seq = torch.cat(use_seq, dim=2)
        loc_seq = self.inp_embedding(loc_seq).transpose(0, 1)
        trans_out = self.transformer_encoder(loc_seq, mask=attn_mask,
                                             src_key_padding_mask=key_padding_mask).transpose(0, 1)
        out = []
        for i in range(nb_batches):
            if self.use_addr:
                addr_inp = torch.cat((addr[i].unsqueeze(0), self.poi_type_embedding(addr_type[i])), dim=1)
                out_i = self.fc_combine(torch.tanh(
                    self.fc_output(trans_out[i][:data_length[i]]) +
                    self.fc_addr(addr_inp).expand(data_length[i], -1))).squeeze(dim=1)
            else:
                out_i = self.fc_combine(torch.tanh(
                    self.fc_output(trans_out[i][:data_length[i]]))).squeeze(dim=1)
            out_i = F.log_softmax(out_i, dim=0)
            out.append(out_i)
        return out

    def generate_key_padding_mask(self, data_length):
        # N * Max_Length
        # False: unchanged True: ignored
        bsz = len(data_length)
        max_len = max(data_length)
        key_padding_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
        for i in range(bsz):
            key_padding_mask[i, data_length[i]:] = True
        return key_padding_mask.to(self.device)

    def generate_attn_mask(self, nb_heads, data_length):
        bsz = len(data_length)
        max_len = max(data_length)
        attn_mask = torch.ones((bsz * nb_heads, max_len, max_len), dtype=torch.bool)
        for i in range(bsz):
            attn_mask[i * nb_heads:(i + 1) * nb_heads, :, :data_length[i]] = False
        return attn_mask.to(self.device)


class LocMatcherPN(nn.Module):
    def __init__(self, hidden_dim, loc_inp_dim):
        super(LocMatcherPN, self).__init__()
        fc_hidden_dim = 32
        self.hidden_dim = hidden_dim
        poi_embed_dim = 3
        self.poi_type_embedding = nn.Embedding(21, poi_embed_dim)
        addr_quant_dim = 1
        addr_inp_dim = addr_quant_dim + poi_embed_dim
        self.fc_addr = nn.Linear(addr_inp_dim, fc_hidden_dim)
        self.loc_dense_dim, self.time_dist_dim = loc_inp_dim
        time_dist_embed_dim = 0
        if self.time_dist_dim != 0:
            time_dist_embed_dim = 3
            self.time_dist_embedding = nn.Linear(self.time_dist_dim, time_dist_embed_dim)
        loc_inp_dim_fused = self.loc_dense_dim + time_dist_embed_dim
        self.lstm = nn.LSTM(loc_inp_dim_fused, self.hidden_dim, batch_first=True, bidirectional=True)
        self.fc_output = nn.Linear(hidden_dim * 2, fc_hidden_dim, bias=False)
        self.fc_combine = nn.Linear(fc_hidden_dim, 1, bias=False)
        self.device = torch.device('cpu')

    def forward(self, addr, addr_type, loc_dense_seq, time_dist_seq, data_length):
        nb_batches = addr.size(0)
        use_seq = []
        if self.loc_dense_dim != 0:
            use_seq.append(loc_dense_seq)
        if self.time_dist_dim != 0:
            use_seq.append(self.time_dist_embedding(time_dist_seq))
        loc_seq = torch.cat(use_seq, dim=2)
        rnn_inp_packed = pack_padded_sequence(loc_seq, data_length, batch_first=True)
        h0 = torch.zeros(2, nb_batches, self.hidden_dim, device=self.device)
        c0 = torch.zeros(2, nb_batches, self.hidden_dim, device=self.device)
        # gru_out: B x T x m
        gru_out_packed, _ = self.lstm(rnn_inp_packed, (h0, c0))
        gru_out_padded, out_len = pad_packed_sequence(gru_out_packed, batch_first=True)
        # B x T x n - > B x T x 1
        out = []
        for i in range(nb_batches):
            addr_inp = torch.cat((addr[i].unsqueeze(0), self.poi_type_embedding(addr_type[i])), dim=1)
            out_i = self.fc_combine(torch.tanh(
                self.fc_output(gru_out_padded[i][:data_length[i]]) +
                self.fc_addr(addr_inp).expand(data_length[i], -1))).squeeze(dim=1)
            out_i = F.log_softmax(out_i, dim=0)
            out.append(out_i)
        return out
