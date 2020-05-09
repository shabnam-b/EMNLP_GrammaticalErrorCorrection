import torch
import torch.nn as nn

from utils import constant

class MLPAttention(nn.Module):
    def __init__(self, dim):
        super(MLPAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_c = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, context, mask=None, attn_only=False):
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        target = self.linear_in(input) # batch x dim
        source = self.linear_c(context.contiguous().view(-1, dim)).view(batch_size, source_len, dim)
        attn = target.unsqueeze(1).expand_as(context) + source
        attn = self.tanh(attn) # batch x sourceL x dim
        attn = self.linear_v(attn.view(-1, dim)).view(batch_size, source_len)

        if mask is not None:
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        weighted_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), dim=1)
        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, weighted_context, attn


class LSTMAttention(nn.Module):

    def __init__(self, input_size, hidden_size, attn_type='soft', coverage=False):
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.coverage = coverage

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)
        
        self.attention_layer = MLPAttention(hidden_size)
        print("Using {} attention for LSTM.".format(attn_type))

    def forward(self, input, hidden, ctx, ctx_mask=None):
        def recurrence(input, hidden):
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # n_b x hidden_dim
            return hy, cy

        input = input.transpose(0,1) # n_step x n_b x hidden_dim
        nsteps, nbatch = input.size(0), input.size(1)
        ctx_len = ctx.size(1)
        
        h_out, h_tilde_out, c_out, attn_out = [], [], [], []
        cov_out = []

        steps = range(nsteps)
        attn = torch.zeros([nbatch, ctx_len]).cuda()
        cov = 0
        for i in steps:
            hidden = recurrence(input[i], hidden)
            hy, cy = hidden
            if self.coverage:
                cov = cov + attn
                cov_out.append(cov)
                h_tilde, c, attn = self.attention_layer(hy, ctx, cov, mask=ctx_mask)
            else:
                h_tilde, c, attn = self.attention_layer(hy, ctx, mask=ctx_mask)
            h_out.append(hy)
            h_tilde_out.append(h_tilde)
            c_out.append(c)
            attn_out.append(attn)
        h_out = torch.cat(h_out, dim=0).view(nsteps, *h_out[0].size())
        h_tilde_out = torch.cat(h_tilde_out, dim=0).view(nsteps, *h_tilde_out[0].size())
        c_out = torch.cat(c_out, dim=0).view(nsteps, *c_out[0].size())
        attn_out = torch.cat(attn_out, dim=0).view(nsteps, *attn_out[0].size())

        for x in [h_out, h_tilde_out, c_out, attn_out]:
            x.transpose_(0,1)

        if self.coverage:
            cov_out = torch.cat(cov_out, dim=0).view(nsteps, *cov_out[0].size())
            cov_out.transpose_(0,1)

        return h_out, h_tilde_out, c_out, attn_out, cov_out, hidden

