
import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers
from models.utils.continual_model import ContinualModel
from utils.per_buffer import PERBuffer
from utils.prompt_pool import PromptPool
from utils.args import *
from utils.ema import EMA_classifier
import math
import copy
import os
import numpy as np
import pickle
from torch.amp import autocast, GradScaler
import math

scaler = GradScaler()

transformers.logging.set_verbosity(50)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='dualp_vit')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--pw',type=float,default=0.5,help='Penalty weight.')
    parser.add_argument('--freeze_clf',type=int,default=0,help='clf freeze flag')
    parser.add_argument('--init_type',type=str,
                        default='unif',
                        # default='default',
                        help='prompt & key initialization')

    return parser

class CPROMPTVIT(ContinualModel):
    NAME = 'cprompt_vit'
    # COMPATIBILITY = ['class-il']
    
    def __init__(self, backbone, loss, args, transform):
        super(CPROMPTVIT, self).__init__(backbone, loss, args, transform)
        self.patch_embed = self.net.patch_embed        
        self.pos_drop = self.net.pos_drop
        self.patch_drop = self.net.patch_drop
        self.norm_pre = self.net.norm_pre
        self.blocks = self.net.blocks
        self.norm = self.net.norm
        self.fc_norm = self.net.fc_norm
        self.head_drop = self.net.head_drop
        self.head = self.net.head
        self.net.requires_grad_(False)
        self.patch_embed.requires_grad_(False)
        self.pos_drop.requires_grad_(False)
        self.patch_drop.requires_grad_(False)
        self.norm_pre.requires_grad_(False)
        self.blocks.requires_grad_(False)
        self.norm.requires_grad_(False)
        self.fc_norm.requires_grad_(False)
        self.head_drop.requires_grad_(False)

        if args.freeze_clf == 0: 
            self.head.requires_grad_(True)
            #torch.nn.init.zeros_(self.classifier.weight)
        else:                    
            self.head.requires_grad_(False)
            
        self.learning_param = None

        self.lr = args.lr
        self.args = args

        # ! cifar100 L2P official param setting
        self.topN = self.args.topN
        self.prompt_num = self.args.prompt_num  # ! equal to 'expert prompt length' in dual prompt paper
        # self.prompt_num = 20  # ! equal to 'expert prompt length' in dual prompt paper
        self.gprompt_num = 4 # ! equal to 'general prompt length' in duap prompt paper
        if self.args.pool_size != -1:
            self.pool_size = self.args.pool_size
        elif self.args.dataset == "seq-cifar100":
            self.pool_size = 10  # ! pool size per layer. if dual prompt : entire pool size = task num
        elif self.args.dataset == "domain-net":
            self.pool_size = 6
        elif self.args.dataset == "imagenet-c":
            self.pool_size = 15
        elif self.args.dataset == "imagenet-r":
            self.pool_size = 15
        elif self.args.dataset == "imagenet-cr":
            self.pool_size = 30
        self.layer_g = [0,1] # layer that attach general prompt : first, second layer by paper
        self.layer_e = [2,3,4] # layer that attach expert prompt : third to fifth layer by paper
        # TODO
        # self.layer_e = [] # layer that attach expert prompt : third to fifth layer by paper
        
        # ! init promptpool
        self.pool = PromptPool()
        self.pool.initPool(12,self.pool_size,self.prompt_num,768,768,self.device,embedding_layer=None,init_type=args.init_type)
        self.general_prompt = [torch.rand((self.gprompt_num,768),requires_grad=True,device=self.device) for i in range(12)] #Lg = 5 in 

        self.init_opt(args)

        # prompt num per task
        self.prompt_per_task = self.args.prompt_per_task
      
        self.prompt_selected_train = []
        self.prompt_selected_sum_train = np.zeros((self.pool_size, ), dtype=int)
        self.prompt_selected_test = []
        self.prompt_selected_sum_test = np.zeros((self.pool_size, ), dtype=int)
        

        # init ema 
        if self.args.use_ema:
            self.backup = copy.deepcopy(self.general_prompt)
            self.general_prompt_ema = copy.deepcopy(self.general_prompt)
            self.decay = self.args.ema_beta
            self.ema_g_flag = True
        
        # init ema for classifier
        if self.args.use_ema_c:
            self.ema_c = EMA_classifier(self, self.args.ema_beta_c, args)
        
        if self.args.adapt_ema_c:
            self.g_before = copy.deepcopy(self.general_prompt[0:2])
            self.cnt = 0
            self.g_change = 0
            self.average_g_change = 0
            self.gamma = 0
            

    
    def init_opt(self,args,task_id=0):
        if self.args.use_prompt_penalty_4 and task_id > 0:
            key_list = []
            prompt_list = []
            for i in range(12):
                key_list += self.pool.key_list[i][task_id * self.prompt_per_task:]
                prompt_list += self.pool.prompt_list[i][task_id * self.prompt_per_task:]
        else:
            key_list = [e for layer_k in self.pool.key_list for e in layer_k]
            prompt_list = [e for layer_p in self.pool.prompt_list for e in layer_p]
        if args.freeze_clf == 0:
            self.learning_param = key_list+prompt_list+list(self.head.parameters())+self.general_prompt
            self.opt = torch.optim.AdamW(params=self.learning_param,lr=self.lr)
        else:
            self.learning_param = key_list+prompt_list+self.general_prompt
            self.opt = torch.optim.AdamW(params=self.learning_param,lr=self.lr)

    # ! untouched
    def similarity(self, pool, q, k, topN, task_id):
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        sim = torch.matmul(q, k.T)  # (B, T)
        if self.args.use_prompt_penalty and self.net.training == True:
            prompt_selected_sum = torch.Tensor(self.prompt_selected_sum_train)
            if self.args.use_gpu:
                prompt_selected_sum = prompt_selected_sum.to('cuda')
            total = torch.sum(prompt_selected_sum)
            if total == 0:
                freq = prompt_selected_sum / 1
            else:
                freq = prompt_selected_sum / total
            dist = 1 - sim
            dist = dist * freq 
        elif self.args.use_prompt_penalty_2 and self.net.training == True:
            prompt_selected_sum = torch.Tensor(self.prompt_selected_sum_train)
            if self.args.use_gpu:
                prompt_selected_sum = prompt_selected_sum.to('cuda')
            total = torch.sum(prompt_selected_sum)
            if total == 0:
                freq = prompt_selected_sum / 1
            else:
                freq = prompt_selected_sum / total
            dist = 1 - sim
            if total <= self.args.pool_size * self.args.batch_size:
                dist = dist * freq   
        elif self.args.use_prompt_penalty_3 and self.net.training == True:
            mask = torch.zeros(self.pool_size)
            if self.args.use_gpu:
                mask = mask.to('cuda')
            mask[task_id * self.prompt_per_task : (task_id+1) * self.prompt_per_task] = 1
            sim = sim * mask
            dist = 1 - sim 
            dist[:,:task_id * self.prompt_per_task] = 2
            dist[:,(task_id+1) * self.prompt_per_task:] = 2
        elif self.args.use_prompt_penalty_4 and self.net.training == True:
            mask = torch.zeros(self.pool_size)
            if self.args.use_gpu:
                mask = mask.to('cuda')
            mask[task_id * self.prompt_per_task : (task_id+1) * self.prompt_per_task] = 1
            dist = 1 - sim * mask

            if task_id > 0:
                mask1 = torch.zeros(self.pool_size)
                if self.args.use_gpu:
                    mask1 = mask1.to('cuda')
                mask1[0 : task_id * self.prompt_per_task] = 1
                dist1 = 1 - sim * mask1
        else:
            dist = 1 - sim

        if self.args.use_prompt_penalty_4 and self.net.training == True and task_id > 0:
            val1, idx1 = torch.topk(dist1, 1, dim=1, largest=False)

            val, idx = torch.topk(dist, topN-1, dim=1, largest=False)

            val = torch.cat((val1, val), dim=1)
            idx = torch.cat((idx1, idx), dim=1)


        else:
            val, idx = torch.topk(dist, topN, dim=1, largest=False)
        dist_pick = []
        for b in range(idx.shape[0]):
            pick = []
            for i in range(idx.shape[1]):
                pick.append(dist[b][idx[b][i]])
            dist_pick.append(torch.stack(pick))

        dist = torch.stack(dist_pick)

        # print("dist:", dist.shape)
        return dist, idx
        
    def getPrompts(self,layer,pool,keys, distance):
        B = keys.shape[0]
        topN = keys.shape[1]

        if layer in self.layer_g:
            prompts = self.general_prompt[layer].unsqueeze(0).repeat(B,1,1)
            return prompts
        
        elif layer in self.layer_e:
            pTensor = torch.stack(pool.prompt_list[layer])
            T, Lp, Dp = pTensor.shape

            # ! selectedKeys: (Batch_size, topN)
            prompts = pTensor[keys,:,:]  # ! (B, topN, Lp, Dp)
            # print("prompts:", prompts.shape)
            if self.args.fuse_prompt or self.args.fuse_prompt_2:
                prompts = torch.mean(prompts, dim=1)
                prompts = prompts.reshape(B,-1,Dp)   # ! (B, topN*Lp, Dp)
                return prompts
            # input()
            elif self.args.prompt_comp:
                half = int(Lp/2)
                prompts_k = prompts[:,:,:half,:].reshape(B, -1, Dp)
                prompts_v = prompts[:,:,half:,:].reshape(B, -1, Dp)
                prompts = torch.cat((prompts_v, prompts_k), dim=1)
                return prompts
            prompts = prompts.reshape(B,-1,Dp)   # ! (B, topN*Lp, Dp)
            return prompts

        else:
            return None
    
    
    def forward_attn(self, Block, x, prompts):
        Attn = Block.attn
        B, N, C = x.shape     
        
        qkv_q = Attn.qkv(x).reshape(B, N, 3, Attn.num_heads, Attn.head_dim).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)
        q, _, _ = qkv_q.unbind(0)
        # print(prompts)
        half = int(prompts.shape[1]/2)
        qkv_k = torch.cat([x[:,0,:].unsqueeze(1), prompts[:, :half ,:] , x[:,1:,:]], 1)
        qkv_k = Attn.qkv(qkv_k).reshape(B, -1, 3, Attn.num_heads, Attn.head_dim).permute(2, 0, 3, 1, 4)
        _, k, _ = qkv_k.unbind(0)
        qkv_v = torch.cat([x[:,0,:].unsqueeze(1), prompts[:, half: ,:] , x[:,1:,:]], 1)
        qkv_v = Attn.qkv(qkv_v).reshape(B, -1, 3, Attn.num_heads, Attn.head_dim).permute(2, 0, 3, 1, 4)
        _, _, v = qkv_v.unbind(0)
        
        q, k = Attn.q_norm(q), Attn.k_norm(k)

        if Attn.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=Attn.attn_drop.p if Attn.training else 0.,
            )
        else:
            q = q * Attn.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = Attn.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = Attn.proj(x)
        x = Attn.proj_drop(x)
        return x
    
    
    def forward_block(self, layer, x, prompts):
        Block = self.net.blocks[layer]
        x = x + Block.drop_path1(Block.ls1(self.forward_attn(Block, Block.norm1(x), prompts)))
        x = x + Block.drop_path2(Block.ls2(Block.mlp(Block.norm2(x))))
        return x
        
    
    def forward_vit(self, x, keys, distance):
        net = self.net
        # x = self.net.forward_features(x)
        x = net.patch_embed(x)
        x = net._pos_embed(x)
        x = net.patch_drop(x)
        x = net.norm_pre(x)
        if net.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(net.blocks, x)
        else:
            # x = net.blocks(x)
            for layer, Block in enumerate(self.net.blocks):
                if layer < self.pool.layer:
                    prompts = self.getPrompts(layer, self.pool, keys, distance)
                if prompts is not None:
                    prompts = prompts.to(x.device)
                    x = self.forward_block(layer, x, prompts)
                else:
                    x = x + Block.drop_path1(Block.ls1(Block.attn(Block.norm1(x))))
                    x = x + Block.drop_path2(Block.ls2(Block.mlp(Block.norm2(x))))
        x = self.norm(x)
        z_clf = x[:, 0, :]
        x = self.net.forward_head(x)
        return x, z_clf

    def forward_cprompt(self, inputs, task_id=None):
        with torch.no_grad():
            representations = self.net.forward_features(inputs)  # [128, 197, 768]
            query = representations[:, 1, :]
        kTensor = torch.stack(self.pool.key_list[0])
        kTensor = kTensor.to(query.device)
        
        if (self.net.training == False or self.args.prompt_comp):
            distance, keys = self.similarity(self.pool,query,kTensor,self.topN,task_id)
        else:
            keys = torch.tensor(task_id,requires_grad=False,device=self.device).unsqueeze(0).repeat(embedding.shape[0],1)
            q = nn.functional.normalize(query,dim=-1)
            k = nn.functional.normalize(kTensor,dim=-1)
            distance = 1-torch.matmul(q,k.T)[:,task_id].unsqueeze(1)
        self.solve_selected_prompt(keys)
        logits, z_clf = self.forward_vit(inputs, keys, distance)
        return logits, distance , z_clf

    # ! util functions for compatibility with other experiment pipeline
    def forward_model(self, x: torch.Tensor, task_id=None):
        logits, distance, z_clf = self.forward_cprompt(x, task_id)
        return logits
        
    def observe(self, inputs, labels,dataset,t):
        with autocast(device_type='cuda'):
            logits, distance, z_clf  = self.forward_cprompt(inputs,t)
            loss = self.loss(logits, labels) + self.args.pw * torch.mean(torch.sum(distance,dim=1))
            self.opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.opt)
            scaler.update()
        return loss.item()

    def cal_g_change(self):
        with torch.no_grad():
            d = 0
            for i in range(2):
                d += torch.sum((self.g_before[i] - self.general_prompt[i]) ** 2)
            d = torch.sqrt(d)
            d = d.item()
            self.cnt += 1
            self.g_before[0] = self.general_prompt[0].clone()
            self.g_before[1] = self.general_prompt[1].clone()
            self.g_change += d
            self.average_g_change = self.g_change / self.cnt
            h = self.args.adapt_h
            self.gamma = 1 - math.exp(d / self.average_g_change - h)

    def update_ema_g(self):
        if self.ema_g_flag:
            self.general_prompt_ema = copy.deepcopy(self.general_prompt)
            self.ema_g_flag = False
        else:
            for i in range(12):
                new_average = (1.0 - self.decay) * self.general_prompt[i].data + \
                                self.decay * self.general_prompt_ema[i]
                self.general_prompt_ema[i] = new_average.clone()

    def ema_before_eval(self, t, w=0.5):
        if self.args.use_ema:
            for i in range(12):
                self.backup[i] = self.general_prompt[i].data
                self.general_prompt[i].data = self.general_prompt_ema[i]
        if self.args.use_ema_c:
            self.ema_c.apply_shadow(w)
    
    def ema_after_eval(self):
        if self.args.use_ema:
            for i in range(12):
                self.general_prompt[i].data = self.backup[i]
        if self.args.use_ema_c:
            self.ema_c.restore()
    
    # save the selected prompts
    def solve_selected_prompt(self, keys):
        keys_c = keys.cpu()
        for key in keys_c:
            if self.net.training is True:
                self.prompt_selected_train.append(key)
                self.prompt_selected_sum_train[key] += 1
            elif self.net.training is False:
                self.prompt_selected_test.append(key)
                self.prompt_selected_sum_test[key] += 1
    
    def save_selected_prompt_to_file(self):
        path = os.path.join(self.args.output_path, self.args.info)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, 'selected_prompts.txt')
        with open(path,'w',encoding='utf-8') as f:
            f.write("sum:\n")
            f.write("train:\n")
            for i, item in enumerate(self.prompt_selected_sum_train):
                f.write(str(item))
                if i % 10 == 9:
                    f.write("\n")
                else:
                    f.write("\t")
            f.write("\ntest\n")
            for i, item in enumerate(self.prompt_selected_sum_test):
                f.write(str(item))
                if i % 10 == 9:
                    f.write("\n")
                else:
                    f.write("\t")
            f.write("\n\nall\ntrain:\n")
            for i, item in enumerate(self.prompt_selected_train):
                f.write(str(item.tolist()))
                if i % 10 == 9:
                    f.write("\n")
                else:
                    f.write("\t")
            f.write("\n\n\n\ntest:\n")
            for i, item in enumerate(self.prompt_selected_test):
                f.write(str(item.tolist()))
                if i % 10 == 9:
                    f.write("\n")
                else:
                    f.write("\t")


    def save_ema_to_file(self, task_id=None):
        if self.args.use_ema_c:
            path = os.path.join(self.args.output_path, self.args.info, 'ema_c')
            if not os.path.exists(path):
                os.makedirs(path)
            path_ema_c = os.path.join(path, str(task_id)+'_ema_c.pkl')
            with open(path_ema_c, 'wb') as f:
                pickle.dump(self.ema_c.shadow, f)
        if self.args.use_ema:
            path = os.path.join(self.args.output_path, self.args.info, 'ema_g')
            if not os.path.exists(path):
                os.makedirs(path)
            path_ema_g = os.path.join(path, str(task_id)+'_ema_g.pkl')
            with open(path_ema_g, 'wb') as f:
                pickle.dump(self.general_prompt_ema, f)

    def load_ema_from_file(self, task_id=None):
        if self.args.use_ema_c:
            path = os.path.join(self.args.output_path, self.args.info, 'ema_c')
            if not os.path.exists(path):
                os.makedirs(path)
            path_ema_c = os.path.join(path, str(task_id)+'_ema_c.pkl')
            with open(path_ema_c, 'rb') as f:
                self.ema_c.shadow = pickle.load(f)
            print("Sucess load the ema_c from: ", path_ema_c)
            
        if self.args.use_ema:
            path = os.path.join(self.args.output_path, self.args.info, 'ema_g')
            if not os.path.exists(path):
                os.makedirs(path)
            path_ema_g = os.path.join(path, str(task_id)+'_ema_g.pkl')
            with open(path_ema_g, 'rb') as f:
                self.general_prompt_ema = pickle.dump(f)
        # if self.args.use_gpu:
        #     self.general_prompt.to('cuda')
        

    def save_prompt_to_file(self, task_id=None):
        path = os.path.join(self.args.output_path, self.args.info, 'prompt_pool')
        if not os.path.exists(path):
            os.makedirs(path)
        path_pool = os.path.join(path, str(task_id)+'_prompt_pool.pkl')
        self.pool.retainPool(path_pool)
        # save the prompt pool
        path = os.path.join(self.args.output_path, self.args.info, 'g_prompt')
        if not os.path.exists(path):
            os.makedirs(path)
        path_g_prompt = os.path.join(path, str(task_id)+'_g_prompt.pt')
        torch.save(self.general_prompt, path_g_prompt)

    def load_prompt_from_file(self, task_id=None):
        path = os.path.join(self.args.output_path, self.args.info, 'prompt_pool')
        if not os.path.exists(path):
            os.makedirs(path)
        path_pool = os.path.join(path, str(task_id)+'_prompt_pool.pkl')
        self.pool.loadPool(path_pool, update_key=True, update_prompt=True, device='cuda')
        # load general prompt
        path = os.path.join(self.args.output_path, self.args.info, 'g_prompt')
        path_g_prompt = os.path.join(path, str(task_id)+'_g_prompt.pt')
        self.general_prompt = torch.load(path_g_prompt)
        # if self.args.use_gpu:
        #     self.general_prompt.to('cuda')
        print("Sucess load the prompt pool from: ", path_pool)
        print("Sucess load the g prompt from: ", path_g_prompt)
    
    def save_model_to_file(self, task_id=None):
        path = os.path.join(self.args.output_path, self.args.info, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        path_model = os.path.join(path, str(task_id)+'model.pt')
        torch.save(self.state_dict(), path_model)

    def load_model_from_file(self, task_id=None):
        path = os.path.join(self.args.output_path, self.args.info, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        path_model = os.path.join(path, str(task_id)+'model.pt')
        self.load_state_dict(torch.load(path_model))
        print("Sucess load the model from ", path_model)

    def set_g_prompt_not_update(self):
        for i in self.general_prompt:
            i.requires_grad = False
            
