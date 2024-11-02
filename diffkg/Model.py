import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
from torch_scatter import scatter_sum, scatter_softmax
import math

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, handler):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.eEmbeds = nn.Parameter(init(torch.empty(args.entity_n, args.latdim)))
		self.rEmbeds = nn.Parameter(init(torch.empty(args.relation_num, args.latdim)))

		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
		self.rgat = RGAT(args.latdim, args.layer_num_kg, args.mess_dropout_rate)

		self.kg_dict = handler.kg_dict
		self.edge_index, self.edge_type = self.sampleEdgeFromDict(self.kg_dict, triplet_num=args.triplet_num)
	
	def getEntityEmbeds(self):
		return self.eEmbeds

	def getUserEmbeds(self):
		return self.uEmbeds
				
	def forward(self, adj, mess_dropout=True, kg=None):
		if kg == None:
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, [self.edge_index, self.edge_type], mess_dropout)
		else:
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, kg, mess_dropout)
						
		embeds = torch.concat([self.uEmbeds, hids_KG[:args.item, :]], axis=0)#拼接用户嵌入和知识图谱嵌入
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)#输入多层gcn并求和叠加
		embeds = sum(embedsLst)

		return embeds[:args.user], embeds[args.user:]#返回用户嵌入，物品嵌入
	
	def sampleEdgeFromDict(self, kg_dict, triplet_num=None):
		sampleEdges = []
		for h in kg_dict:
			t_list = kg_dict[h]#头实体对应的边数
			if triplet_num != -1 and len(t_list) > triplet_num:
				sample_edges_i = random.sample(t_list, triplet_num)#确保采样数量为triplet_num
			else:
				sample_edges_i = t_list
			for r, t in sample_edges_i:
				sampleEdges.append([h, t, r])
		return self.getEdges(sampleEdges)
	
	def getEdges(self, kg_edges):
		graph_tensor = torch.tensor(kg_edges)
		index = graph_tensor[:, :-1]#三元组转换为张量，并分离出边的索引和类型
		type = graph_tensor[:, -1]
		return index.t().long().cuda(), type.long().cuda()
	
class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)
		
class RGAT(nn.Module):
	def __init__(self, latdim, n_hops, mess_dropout_rate=0.4):
		super(RGAT, self).__init__()
		self.mess_dropout_rate = mess_dropout_rate
		self.W = nn.Parameter(init(torch.empty(size=(2*latdim, latdim)), gain=nn.init.calculate_gain('relu')))

		self.leakyrelu = nn.LeakyReLU(0.2)
		self.n_hops = n_hops
		self.dropout = nn.Dropout(p=mess_dropout_rate)

	def agg(self, entity_emb, relation_emb, kg):
		edge_index, edge_type = kg
		head, tail = edge_index
		a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)#拼接首尾嵌入
		e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)#实体嵌入与关系嵌入逐元素相乘，再在最后一个维度求和
		e = self.leakyrelu(e_input)
		e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
		agg_emb = entity_emb[tail] * e.view(-1, 1)
		agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
		agg_emb = agg_emb + entity_emb#注意力机制融合
		return agg_emb
		
	def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
		entity_res_emb = entity_emb
		for _ in range(self.n_hops):
			entity_emb = self.agg(entity_emb, relation_emb, kg)
			if mess_dropout:
				entity_emb = self.dropout(entity_emb)
			entity_emb = F.normalize(entity_emb)

			entity_res_emb = args.res_lambda * entity_res_emb + entity_emb#结果与原先嵌入加权求和
		return entity_res_emb
	
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]#时间步乘以频率
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)#通过余弦和正弦拼接得到时间嵌入
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)#维度为奇数添加0向量
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)#拼接嵌入与时间嵌入
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)#在输入线性层传播并使用tanh激活
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:#在输出线性层传播并使用tanh激活，除最后一层
				h = torch.tanh(h)

		return h

class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)#step个值的限制范围数组以随机化方差
		alpha_bar = 1 - variance#控制噪声强度的alpha值
		betas = []
		betas.append(1 - alpha_bar[0])#添加第一个alpha
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))#通过公式计算beta数组，表示噪声强度
		return np.array(betas)
	
	def calculate_for_diffusion(self):#计算
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()#从初始状态到当前步骤的信号强度的累积效果
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()#self.alphas_cumprod 向前移动一位，并在开头添加一个值为 1.0 的元素
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()#self.alphas_cumprod 向后移动一位，并在末尾添加一个值为 0.0 的元素。

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)#累积信号强度的平方根
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (#后验方差
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))#后验均值系数1
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))#后验均值系数2

	def p_sample(self, model, x_start, steps):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]#反向步数

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()#创建一个与当前批次样本数量相同的张量 t，并将其填充为当前时间步 i 的值，还是用来表示时间步
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			x_t = model_mean
		return x_t
			
	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
	#加噪过程
	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):#匹配形状
		arr = arr.cuda()
		res = arr[timesteps].float()#取arr对应的浮点数，用来获取sqrt_alphas_cumprod与sqrt_one_minus_alphas_cumprod的对应时间步值
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)#增加维度，匹配形状
	
	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)#实际用到是Denoisemodel，其中将x嵌入融合时间步信息

		model_variance = self.posterior_variance#后验方差赋值，在beta赋值时就定义好了
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)#对齐
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		#这里使用 _extract_into_tensor 方法提取与时间步 t 对应的均值系数，并与模型输出和输入 x 进行加权求和
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, ui_matrix, userEmbeds, itmEmbeds, batch_index):
		batch_size = x_start.size(0)
		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()#ts 表示在扩散过程中选择的时间步，用于后续的噪声添加或样本生成。
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, ts)
		mse = self.mean_flat((x_start - model_output) ** 2)#均方误差

		weight = self.SNR(ts - 1) - self.SNR(ts)#当前时间步与前一时间步信噪比差异，作为损失权重
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse#差异损失

		item_user_matrix = torch.spmm(ui_matrix, model_output[:, :args.item].t()).t()#ui_matrix是原始交互矩阵，而item_user_matrix是将output的物品特征表示与原始矩阵作矩阵乘法所得
		itmEmbeds_kg = torch.mm(item_user_matrix, userEmbeds)#在得物品特征表示
		ukgc_loss = self.mean_flat((itmEmbeds_kg - itmEmbeds[batch_index]) ** 2)#物品嵌入与当前批次物品嵌入均方误差

		return diff_loss, ukgc_loss
		
	def mean_flat(self, tensor):#忽略第一个维度的均值
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):#信噪比
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])