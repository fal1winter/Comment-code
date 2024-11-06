import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE

# Paper: XSimGCL - Towards Extremely Simple Graph Contrastive Learning for Recommendation


class XSimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCL, self).__init__(conf, training_set, test_set)
        config = self.config['XSimGCL']
        self.cl_rate = float(config['lambda'])
        self.eps = float(config['eps'])
        self.temp = float(config['tau'])
        self.n_layers = int(config['n_layer'])
        self.layer_cl = int(config['l_star'])
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):#遍历一个生成器函数next_batch_pairwise返回的批次数据
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = model(True)#调用model函数，并传入参数True，该函数返回推荐用户嵌入（rec_user_emb）、推荐物品嵌入（rec_item_emb）、对比用户嵌入（cl_user_emb）和对比物品嵌入（cl_item_emb）
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]#提取嵌入向量
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)#双损失计算
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class XSimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)#将用户嵌入和项目嵌入按行拼接
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):#遍历每一层
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)#计算得到当前层的嵌入表示
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()#生成随机噪声，由区间[0,1)上均匀分布的随机数填充
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps#eps为常数，用于控制更新的幅度，通常用于防止数值不稳定或避免过拟合，sign返回ego每个元素的符号
            all_embeddings.append(ego_embeddings)#当前层添加到all
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings#最后一层用于进行cl
        final_embeddings = torch.stack(all_embeddings, dim=1)#将 all_embeddings 中的所有嵌入向量沿着指定的维度（这里是维度1）堆叠在一起，形成一个新的张量 final_embeddings
        final_embeddings = torch.mean(final_embeddings, dim=1)#将 final_embeddings 按照用户数量和项目数量进行拆分，分别得到用户的嵌入向量 user_all_embeddings 和项目的嵌入向量 item_all_embeddings
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:#是否返回扰动
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings
