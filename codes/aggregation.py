import torch.nn as nn
import torch

# relation aggregation\
# class RelationAggregate(nn.Module):
#
#     def __init__(self, instance_dim, ontology_dim):
#         super(RelationAggregate, self).__init__()
#         self.attention_weight = nn.Linear(instance_dim + ontology_dim, 1)
#         self.transfor = nn.Linear(ontology_dim * 3, instance_dim)
#
#     def forward(self, r_in, r_on, h_on, t_on):
#         triple_on_agg = r_on * h_on * t_on
#         entity_on_agg = h_on * t_on
#         r_on_agg = r_in * r_on
#         agg = self.transfor(torch.cat((triple_on_agg, entity_on_agg, r_on_agg), dim=-1))
#         epsilon = self.attention_weight(torch.cat((agg, r_on_agg), dim=-1))
#         epsilon = torch.sigmoid(epsilon)
#         r = epsilon * r_in + (1 - epsilon) * agg
#         return r
from torch.nn import Parameter


class MoE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MoE, self).__init__()
        self.experts_h = nn.Linear(input_size, hidden_size)
        self.experts_r = nn.Linear(input_size, hidden_size)
        self.experts_t = nn.Linear(input_size, hidden_size)
        self.gating = nn.Linear(input_size, 1)

    def forward(self, r_in, r_on, h_on, t_on):
        h = torch.stack([h_on, r_on, t_on], dim=1).squeeze(2)  # batch 3 hidden_size
        p = self.gating(h)  # batch num_expert
        hh = self.experts_h(h_on)
        rh = self.experts_r(r_on)
        th = self.experts_t(t_on)
        # e = torch.matmul(h, self.experts_w.permute(2, 1, 0))  # batch num_expert hidden_size
        y = torch.bmm(p.permute(0, 2, 1), torch.stack([hh, rh, th], dim=1).squeeze(2))
        return y


class AggregateR(nn.Module):

    def __init__(self, instance_dim, ontology_dim):
        super(AggregateR, self).__init__()
        self.instance_dim = instance_dim
        self.ontology_dim = ontology_dim
        # self.attention_weight = nn.Linear(instance_dim + ontology_dim, 1,)
        # self.transfor = nn.Linear(ontology_dim * 3, instance_dim)

    def forward(self, r_in, r_on, h_on, t_on):
        r = r_on
        return r


class AggregateTR(nn.Module):

    def __init__(self, instance_dim, ontology_dim):
        super(AggregateTR, self).__init__()
        self.instance_dim = instance_dim
        self.ontology_dim = ontology_dim
        # self.attention_weight = nn.Linear(instance_dim + ontology_dim, 1,)
        self.transfor = nn.Linear(ontology_dim, instance_dim)

    def forward(self, r_in, r_on, h_on, t_on):
        r = self.transfor(r_on)
        return r


class AggregateWeight(nn.Module):

    def __init__(self, instance_dim, ontology_dim):
        super(AggregateWeight, self).__init__()
        self.instance_dim = instance_dim
        self.ontology_dim = ontology_dim
        # self.attention_weight = nn.Linear(instance_dim + ontology_dim, 1,)
        self.weight = nn.Parameter(torch.Tensor([0.3]), requires_grad=False)
        self.transfor = nn.Linear(ontology_dim, instance_dim)

    def forward(self, r_in, r_on, h_on, t_on):
        agg = self.transfor(r_on)
        r = (1 - self.weight) * r_in + self.weight * agg
        return r


class AggregateSoft(nn.Module):

    def __init__(self, instance_dim, ontology_dim):
        super(AggregateSoft, self).__init__()
        self.instance_dim = instance_dim
        self.ontology_dim = ontology_dim
        self.attention_weight = nn.Linear(instance_dim + ontology_dim, 1, )
        self.transfor = nn.Linear(ontology_dim, instance_dim)

    def forward(self, r_in, r_on, h_on, t_on):
        agg = self.transfor(r_on)
        epsilon = self.attention_weight(torch.cat((r_in, agg), dim=-1))
        epsilon = torch.sigmoid(epsilon)
        r = epsilon * r_in + (1 - epsilon) * agg
        return r


class AggregateTriple(nn.Module):

    def __init__(self, instance_dim, ontology_dim):
        super(AggregateTriple, self).__init__()
        self.instance_dim = instance_dim
        self.ontology_dim = ontology_dim
        self.transfor = nn.Linear(ontology_dim * 3, instance_dim)

    def forward(self, r_in, r_on, h_on, t_on):
        agg = self.transfor(torch.cat((r_on, h_on, t_on), dim=-1))
        return agg


class AggregateHR(nn.Module):
    def __init__(self, instance_dim, ontology_dim):
        super(AggregateHR, self).__init__()
        self.instance_dim = instance_dim
        self.ontology_dim = ontology_dim
        self.transfor = nn.Linear(ontology_dim * 2, instance_dim)

    def forward(self, r_in, r_on, h_on, t_on):
        agg = self.transfor(torch.cat((r_on, h_on), dim=-1))
        return agg


class AggregateEntity(nn.Module):

    def __init__(self, instance_dim, ontology_dim):
        super(AggregateEntity, self).__init__()
        self.instance_dim = instance_dim
        self.ontology_dim = ontology_dim
        self.transfor = nn.Linear(ontology_dim * 2, instance_dim)

    def forward(self, r_in, r_on, h_on, t_on):
        agg = self.transfor(torch.cat((h_on, t_on), dim=-1))
        return agg


class GRL(nn.Module):

    def __init__(self, instance_dim, relation_class):
        super(GRL, self).__init__()
        self.K = relation_class
        self.M = nn.Parameter(torch.randn(self.K, instance_dim))

        self.instance_dim = instance_dim
        # self.transfor = nn.Linear(instance_dim, instance_dim)
        self.j_FC = nn.Linear(instance_dim, instance_dim)
        self.classer = nn.Linear(instance_dim, relation_class)
        self.r_loss = nn.CrossEntropyLoss()

    def forward(self, h_on, t_on, relation_lable=None, is_train=True):
        j = h_on - t_on  # (1,dim)
        pf = self.j_FC(j)
        pf = torch.sigmoid(pf)  # (1,dim)
        a_sim = torch.softmax(torch.matmul(j, self.M.t()), dim=-1)  # (dim) (dim ,K) = (1,K)
        if is_train:
            rk = torch.matmul(a_sim, self.M)  # (1,dim)
            f = (1 - pf) * j + pf * rk  # (1,dim)
            f = self.classer(f).squeeze(1)
            loss = self.r_loss(f, relation_lable)
            return loss, torch.argmax(f, dim=1)
        else:
            rk = torch.matmul(a_sim, self.M)  # (1,dim)
            f = (1 - pf) * j + pf * rk  # (1,dim)
            f = self.classer(f).squeeze(1)
            q, q_idx = torch.max(f, dim=2)
            w, w_idx = torch.max(q, dim=1)
            B = q_idx[[i for i in range(f.shape[0])], w_idx]
            print(B)
            # return B
            out = self.M[B]
            return B


if __name__ == '__main__':
    moe = MoE(10, 10)
    r_in = torch.randn(8, 1, 10)
    h_on = torch.randn(8, 1, 10)
    r_on = torch.randn(8, 1, 10)
    t_on = torch.randn(8, 1, 10)
    y = moe(r_in, h_on, r_on, t_on)
    print(y.shape)
    print(y)
