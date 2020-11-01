import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import utils as nn_utils


class GloveMean(nn.Module):
    def __init__(self, args, pretrained_weight):
        super(GloveMean, self).__init__()
        vocab_size, embedding_size = pretrained_weight.shape
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        
    def mean_pool(self, x, x_mask):
        x_num = torch.sum(x_mask, dim=1).unsqueeze(-1).expand(-1, x.size(2))  # (batch, embed_size)
        x_sum = torch.sum(x, dim=1)
        s = x_sum/(x_num+1e-20)
        return s
        
    def forward(self, x, x_mask):  # x:(batch, max_sent_len)
        x = self.embed(x)          # (batch, max_sent_len, embed_size)
        s = self.mean_pool(x, x_mask)
        return s
    
class DVAE(nn.Module):
    def __init__(self, args):
        super(DVAE, self).__init__()
        self.K = args.K
        self.M = args.M
        self.alpha = args.alpha
        self.temperature = args.temperature
        self.linear1 = nn.Linear(args.sent_size, args.K*args.M)      # map state to z
        self.linear2 = nn.Linear(args.M, args.sent_size*2, bias=False) # map z to interaction representation
        self.linear3 = nn.Linear(args.M, args.sent_size*2, bias=False)           # map z and signal to context representation
        self.linear4 = nn.Linear(args.sent_size*4, args.sent_size)   # reduce the dimension of interaction+context to sent size
        self.prior_z = nn.Parameter(torch.ones(args.M, args.K)/args.K)  # train prior (M, K)
        
    def gumbel_softmax(self, x):
        u = torch.rand_like(x)
        nn.init.uniform_(u, a=0, b=1)
        g = -torch.log(-torch.log(u + 1e-20) + 1e-20)/300
        temp = (x + g)/self.temperature
        w = F.softmax(temp, dim=2)
        return w
        
    def kl_loss_dvae(self, q, prior):
        if prior.dim() <= 2:
            prior = prior.unsqueeze(0).expand(q.size(0), -1, -1)  # (batch, M, K)
        kl_divergence = torch.sum(q*(torch.log(q+1e-20)-torch.log(prior+1e-20)), dim=2)
        kl_divergence = torch.sum(kl_divergence, dim=1)
        kl_divergence = torch.mean(kl_divergence)
        return kl_divergence
    
    def l2_loss(self, generate, target):
        temp = torch.pow(generate-target, 2)
        re_loss = torch.mean(torch.sum(torch.pow(generate-target, 2), dim=1))
        return re_loss
    
    def cross_entropy_loss(self, generate, target):
        generate = F.softmax(generate, dim=1)
        target = F.softmax(target, dim=1)
        re_loss = torch.mean(-generate*torch.log(target+1e-20), dim=1)
        re_loss = torch.mean(re_loss)
        return re_loss
    
    def dvae(self, x, x_parent):
        state = x
        signal = torch.cat((x, x_parent), dim=1)  # (batch, sent_size*2)        
        l = self.linear1(state)                   # (batch, M*K)
        l = l.reshape(-1, self.M, self.K)         # (batch, M, K)
        #q = F.log_softmax(l, 2)                   # (batch, M, K)
        q = torch.tanh(l)
        q = F.log_softmax(q, 2)                   # (batch, M, K)
        w = self.gumbel_softmax(q)                # (batch, M, K) Gumbel softmax [p(z|state)]
        w_max, w_id = torch.max(w, dim=2)         # (batch, M)
        w_id = w_id.float()+1
        interaction = self.linear2(w_id)/300      # (batch, sent_size)
        
        context = torch.sigmoid(self.linear3(w_id))*signal        # (batch, signal_size)
        #print(context)
        s = self.linear4(torch.cat((interaction, context), dim=1)) # (batch, sent_size)
        prior = F.softmax(self.prior_z, dim=1)  # 1. same for every x (better)
        kl_divergence = self.kl_loss_dvae(w, prior)
        re_loss = self.cross_entropy_loss(s, state)
        loss_vae = torch.abs(re_loss - self.alpha*kl_divergence)
        return s, (loss_vae.item(), re_loss.item(), kl_divergence.item()), w_id
        
    def forward(self, x, adjacency_list):
        parent_index = adjacency_list[:,0]
        x_parent = x[parent_index]
        s, loss_list, z = self.dvae(x, x_parent)
        return s, loss_list, z
    
    
class Evolution(nn.Module):
    def __init__(self, args):
        super(Evolution, self).__init__()
        self.seq1 = nn.LSTM(input_size=args.inter_size+args.sent_size,
                            num_layers=args.num_layers,
                            hidden_size=args.hidden_size,
                            batch_first=True, bidirectional=True)
        self.seq2 = nn.LSTM(input_size=args.hidden_size*2,
                            num_layers=args.num_layers,
                            hidden_size=args.hidden_size,
                            batch_first=True, bidirectional=True)
        
    def batch_seq(self, x, x_mask, seq_func):
        mask_len = torch.sum(x_mask, dim=1)
        _, idx_sort = torch.sort(mask_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        len_sort = list(mask_len[idx_sort])
        x = x.index_select(0, Variable(idx_sort))  # (batch, max_mask_len, sent_size)
        x_pack = nn_utils.rnn.pack_padded_sequence(x, len_sort, batch_first=True)
        out, h_c = seq_func(x_pack, None) 
        out = nn_utils.rnn.pad_packed_sequence(out, batch_first=True) 
        out = out[0].index_select(0, Variable(idx_unsort)) 
        return out
    
    def split_cascade(self, x, edge_order):
         ## recognize source tweet whose edge_order=-1 
        source_index = [i for i in range(len(edge_order)) if edge_order[i]==-1] + [len(edge_order)]
        ## group indexes according to cascade
        cas_index = [list(range(source_index[i], source_index[i+1])) for i in range(len(source_index)-1)]
        ## split sentences into groups of cascades
        x_split = [x[index_list] for index_list in cas_index]
        split_list = [len(item) for item in cas_index]
        max_split_len = max(split_list)
        ## padding to make sure cascades have same length ang stack into tensor
        x_split_pad = [F.pad(item, (0, 0, 0, max_split_len-item.size(0)), 'constant', 0) for item in x_split]
        x_split_pad = torch.stack(tuple(x_split_pad), dim=0)    # (batch_cas, max_cas_size, sent_size)
        x_mask = torch.Tensor([[1]*item+[0]*(max_split_len-item) for item in split_list]).float().to(x.device)
        return x_split_pad, x_mask, split_list
    
        
    def forward(self, x, adjacency_list, node_order, edge_order):
        """
        x: (batch_sent, inter_size)
        adjacency_list: (batch_sent, 2)   [[parent_index, child_index], ...]
        node_order: (batch_sent, )
        edge_order: (batch_sent, )        
        """        
        x_split_pad, x_mask, split_list = self.split_cascade(x, edge_order)  # (batch_cas, max_cas_size, input_size)  (batch_cas, max_cas_size)
        batch_cas = x_split_pad.size(0)
        ## update representation with sequential model
        layer1 = self.batch_seq(x_split_pad, x_mask, self.seq1)  # (batch_cas, max_cas_size, hidden*2)        
        layer2 = self.batch_seq(layer1, x_mask, self.seq2)       # (batch_cas, max_cas_size, hidden*2) 
        ## cut and splice to batch of sentences
        sent = [layer1[i][:split_list[i]] for i in range(batch_cas)]
        sent = torch.cat(tuple(sent), dim=0)                     # (batch_sent, hidden*2)                            
        return sent, layer2, x_mask

# integrate_model = Evolution(args)
# integrate_model.to(device)
# t4 = integrate_model(t2, x2, x3, x4)
# print("integrate")

class Verify(nn.Module):
    def __init__(self, args):
        super(Verify, self).__init__()
        self.attn_m = nn.Linear(args.hidden_size*2, args.hidden_size*2)
        self.attn_u = nn.Linear(args.hidden_size*2, 1, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.hidden_size*2, 3)
        
    def attention_pool(self, x, x_mask, linear1, linear2):
        temp = linear1(x)            # (batch, window_len, hidden*2)
        m = torch.tanh(temp)             # (batch, window_len, hidden*2)
        temp = linear2(m).squeeze(2) # (batch, window_len)
        temp = torch.exp(temp)           # (batch, window_len)
        temp = temp*x_mask.float()       # (batch, window_len)
        temp_sum = torch.sum(temp, dim=1).unsqueeze(-1).expand(-1, x.size(1)) # (batch, window_len)
        u = temp/temp_sum                # (batch, window_len)
        u_ = u.unsqueeze(-1).expand(-1, -1, x.size(2))  # (batch, window_len, hidden*2)
        s = torch.sum(x*u_, dim=1)        # (batch, hidden*2) 
        return s, u
    
    def forward(self, cas, cas_mask):
        tree, weights = self.attention_pool(cas, cas_mask, self.attn_m, self.attn_u)
        tree = self.dropout(tree)
        yvp = self.out(tree)
        return yvp
    
class Stance(nn.Module):
    def __init__(self, args):
        super(Stance, self).__init__()
        self.linear = nn.Linear(args.hidden_size*2, 4) 
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        ysp = self.linear(x)
        return ysp
    
    
class Hierarchy(nn.Module):
    def __init__(self, sent_module, interaction_module, evolution_module, stance_module, verify_module, args, pretrained_weight):
        super(Hierarchy, self).__init__()
        self.sent_model = sent_module(args, pretrained_weight)
        self.interaction_model = interaction_module(args)
        self.evolution_model = evolution_module(args)        
        ## convenient to set different learning rate
        ## because data volume for verification and stance classification differs a lot
        self.verify = verify_module(args)
        self.stance = stance_module(args)
    
    def forward(self, batch_data):
        x, x_mask, adjacency_list, node_order, edge_order, yv, ys = batch_data
        ## represent sentence
        x_sent = self.sent_model(x, x_mask)                                     # (batch_sent, sent_size)
        ## interaction modeling
        x_inter, loss_list, z = self.interaction_model(x_sent, adjacency_list)  # (batch_sent, inter_size)
        #nodes = x_inter
        nodes = torch.cat((x_sent, x_inter), dim=1)
        ## evolution capturing
        sent, cas, cas_mask = self.evolution_model(nodes, adjacency_list, node_order, edge_order) # (batch_sent, hidden*2) (batch_cas, max_cas_size, hidden*2)  (batch_cas, max_cas_size)
        ysp = self.stance(sent)
        yvp = self.verify(cas, cas_mask)
        y_list = (yvp, yv, ysp, ys)
        return y_list, loss_list, z