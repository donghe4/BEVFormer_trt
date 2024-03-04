import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_
import math

class Channel_Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, device=torch.device('cuda'))
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        # x = x.cpu()
        x = self.ln(x)
        # x = x.cuda()
        x = x.permute(0, 3, 1, 2)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,1,device=torch.device('cuda'))
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,1,device=torch.device('cuda'))
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        # x = x.cpu()
        x = self.fc1(x)
        # x = x.cuda()
        x = self.act(x)
        x = self.drop(x)
        # x = x.cpu()
        x = self.fc2(x)
        # x = x.cuda()
        x = self.drop(x)
        return x
    
class NeighborhoodAttention(nn.Module): #It can only use static size as input,but you can define a new input size if you wish.
    def __init__(self,input_size, dim, num_heads,window_size=7, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert window_size%2 == 1,'windowsize must be odd.'
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim,dim*3,1, bias=qkv_bias,device=torch.device('cuda'))
        self.proj = nn.Conv2d(dim, dim, 1,device=torch.device('cuda'))
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.pad_idx = nn.ReplicationPad2d(self.window_size//2)
        self.relative_bias = nn.Parameter(torch.zeros((2*self.window_size-1)**2,num_heads)).cuda()
        trunc_normal_(self.relative_bias, std=.02)
        self.idx_h = torch.arange(0,window_size).cuda()
        self.idx_w = torch.arange(0,window_size).cuda()
        self.idx_k = ((self.idx_h.unsqueeze(-1) * (2*self.window_size-1)) + self.idx_w).view(-1)
        self.set_input_size(input_size)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def attention(self,x):
        B,C,H,W = x.shape
        assert H >= self.window_size and W >= self.window_size,'input size must not be smaller than window size'
        # x= x.cpu()
        qkv = self.qkv(x).view(B, 3,self.num_heads,C//self.num_heads,H*W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        # x = x.cuda()
        attn = q.unsqueeze(3) @ k[:,:,self.attn_idx].transpose(-1,-2) #B,nh,L,1,K^2
        attn = attn + self.relative_bias[self.bias_idx].permute(2, 0, 1).unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v[:,:,self.attn_idx]).squeeze(3).transpose(-1,-2).contiguous().view(B,C,H,W)
        return x
        
    def get_bias_idx(self,H,W):
        num_repeat_h = torch.ones(self.window_size,dtype=torch.long).cuda()
        num_repeat_w = torch.ones(self.window_size,dtype=torch.long).cuda()
        num_repeat_h[self.window_size//2] = H-(self.window_size-1)
        num_repeat_w[self.window_size//2] = W-(self.window_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.window_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        return bias_idx.view(-1,self.window_size**2)
    
    def get_attn_idx(self,H,W):
        H_ = H - (self.window_size - 1)
        W_ = W - (self.window_size - 1)
        attn_idx = torch.arange(0,H_*W_,dtype=torch.float).view(1,1,H_,W_).cuda()
        attn_idx = self.pad_idx(attn_idx).view(-1).type(torch.long)
        attn_idx = self.get_unfold_idx(H,W)[attn_idx]
        return attn_idx
    
    def get_unfold_idx(self,H,W):
        H_ = H-(self.window_size-1)
        W_ = W-(self.window_size-1)
        h_idx = torch.arange(W_).repeat(H_).cuda()
        w_idx = torch.arange(H_).repeat_interleave(W_).cuda() * W
        k_idx_1 = torch.arange(self.window_size).repeat(self.window_size).cuda()
        k_idx_2 = torch.arange(self.window_size).repeat_interleave(self.window_size).cuda() * W
        k_idx = k_idx_1 + k_idx_2
        hw_idx = h_idx + w_idx
        unfold_idx = hw_idx[:,None] + k_idx
        return unfold_idx
    
    def set_input_size(self,input_size):
        H,W = input_size
        self.H,self.W = H,W
        assert H >= self.window_size and W >= self.window_size,'input size must not be smaller than window size'
        attn_idx = self.get_attn_idx(H,W)
        bias_idx = self.get_bias_idx(H,W)
        self.register_buffer("attn_idx", attn_idx)
        self.register_buffer("bias_idx",bias_idx)


        
class NATLayer(nn.Module):
    def __init__(self,input_size, dim, num_heads,window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=Channel_Layernorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(input_size, dim, num_heads,window_size,qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x).cuda()
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def set_input_size(self,input_size):
        self.attn.set_input_size(input_size)


class CrossNeighborhoodAttention(NeighborhoodAttention):
    
    def __init__(self,input_size, dim, num_heads,window_size=7, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__(input_size, dim, num_heads,window_size=7, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
        assert window_size%2 == 1,'windowsize must be odd.'
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Conv2d(2500, dim, 1, bias=qkv_bias,device=torch.device('cuda'))
        # self.q = nn.Linear(8, 256, device=torch.device('cuda'))
        self.kv = nn.Conv2d(375, dim, 1, bias=qkv_bias,device=torch.device('cuda'))
        self.proj = nn.Conv2d(dim, dim, 1, device=torch.device('cuda'))
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.pad_idx = nn.ReplicationPad2d(self.window_size//2)
        self.relative_bias = nn.Parameter(torch.zeros((2*self.window_size-1)**2,num_heads)).cuda()
        self.q_positions = nn.Parameter(torch.randn((dim, 256))).cuda()
        self.v_positions = nn.Parameter(torch.randn((375, 256))).cuda()
        trunc_normal_(self.relative_bias, std=.02)
        self.idx_h = torch.arange(0,window_size).cuda()
        self.idx_w = torch.arange(0,window_size).cuda()
        self.idx_k = ((self.idx_h.unsqueeze(-1) * (2*self.window_size-1)) + self.idx_w).view(-1)
        self.attn = nn.MultiheadAttention(dim , num_heads, dropout=0.,device=torch.device('cuda'))
        self.set_input_size(input_size)


    # def positional_encoding(self, inputs):
    #     B,C,H,W =inputs.shape
    #     y_embed = inputs.cumsum(1, dtype=torch.float32)
    #     x_embed = inputs.cumsum(2, dtype=torch.float32)
    #     eps = 1e-6
    #     y_embed = (y_embed-0.5) / (y_embed[:, -1:, :] + eps)
    #     x_embed = (x_embed-0.5) / (x_embed[:, :, -1:] + eps)
    #     pos = torch.stack([x_embed, y_embed], dim=-1)
    #     bases = [i+1 for i in range(C//2)]

    #     sin = [torch.sin(i * math.pi * pos) for i in bases]
    #     cos = [torch.cos(i * math.pi * pos) for i in bases] 
    #     out = torch.stack((sin+ cos), dim =-1)

    #     # y = torch.linspace(-1.0, 1.0, H).view(-1, 1).repeat(1, W)
    #     # x = torch.linspace(-1.0, 1.0, W).repeat(H, 1)
    #     # pos = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    #     # pos = pos.permute(0, 3, 1, 2)  # Adjust dimensions to [B, 2, H, W]
    #     # pos = pos.flatten(2).permute(0, 2, 1)  # Flatten H, W. Now pos shape is [B, H*W, 2]

    #     # bases = torch.linspace(0, 1, C//2)  # Create bases for sin and cos transformations
    #     # out = torch.cat([torch.sin(2 * math.pi * pos * base) for base in bases] +
    #     #                 [torch.cos(2 * math.pi * pos * base) for base in bases], dim=-1)

    #     # out = out.permute(0, 2, 1)  # Now out shape is [B, C, H*W]
    #     # out = out.view(B, C, H, W)  # Reshape out to [B, C, H, W]

    #     assert not torch.isnan(out).any()
    #     return out

    def forward(self, x, y):
        shortcut, x = self.attention(x, y)
        x = self.proj(x)
        x = self.proj_drop(x)
        return shortcut, x

    # def attention(self, x, y):
    #     B, C, _, _ = x.shape
    #     _, _, H, W = y.shape
    #     assert H >= self.window_size and W >= self.window_size, 'input must not be smaller than window size'

    #     x = x.contiguous().view(6,-1,8)
    #     x = self.q(x).view(B, self.num_heads, C//self.num_heads, H * W).permute(0,1,3,2) # BEV feraure query
    #     kv = self.kv(y).view(B, self.num_heads, C//self.num_heads, H * W).permute(0,1,3,2) # 6 camera feature (encoder output)

    #     q = x * self.scale

    #     attn = q.unsqueeze(3) @ kv[:, :, self.attn_idx].transpose(-1, -2)  # B, nh, L, 1, K^2
    #     attn = attn + self.relative_bias[self.bias_idx].permute(2, 0, 1).unsqueeze(2)
       
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)
    #     qkv = (attn @ kv[:, :, self.attn_idx]).squeeze(3).transpose(-1, -2).contiguous().view(B, C, H, W)
    #     return x.contiguous().view(B,C,H,W) , qkv
    
    # Knn attn algorithm
    def attention(self, query, value): 
        B, C, H, W = query.shape
        _, C_v, _, _ = value.shape
        assert H >= self.window_size and W >= self.window_size, 'input must not be smaller than window size'

        # query = query.view(B,C_q,H*W)
        # value = value.view(B,C_v,H*W)

        # Position Embedding 추가
        q_pos_embedding =self.q_positions.contiguous().view(C,H,W)
        q = query
        # vpos_embedding = self.v_positions.view(C_v,H,W)
        v = value
        # v = value + vpos_embedding

        # q = self.q(q).view(B, self.num_heads, C_q//self.num_heads, H * W).permute(0,1,3,2) # BEV feraure query
        # v = self.v(v).view(B, self.num_heads, C_q//self.num_heads, H * W).permute(0,1,3,2) # 6 camera feature (encoder output)

        q = self.q(q).contiguous().view(B,C,H*W).permute(2,0,1)
        v = self.kv(v).contiguous().view(B,C,H*W).permute(2,0,1)
        # attn = (q @ v.transpose(-2, -1)) * self.scale
        # # # the core code block
        # # mask=torch.zeros(B,self.num_heads, H*W, H*W, device=x.device, requires_grad=False)
        # # index=torch.topk(attn,k=10,dim=-1,largest=True)[1]
        # # mask.scatter_(-1,index,1.)
        # # attn=torch.where(mask>0,attn,torch.full_like(attn,float('-inf')))
        # # # end of the core code block

        # attn = attn.softmax(dim=-1) 
        # attn = self.attn_drop(attn)

        # out = (attn @ v).transpose(3, 2).contiguous().view(B, C, H, W)

        out, _ = self.attn(q, v, v)
        out = out.permute(1, 2, 0).contiguous().view(B, C, H, W)  # rearrange output to match original shape

        return query ,out

class CrossNATLayer(NATLayer):
    def __init__(self,input_size, dim, num_heads,window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=Channel_Layernorm, layer_scale=None):
        super().__init__(input_size, dim, num_heads, window_size, qkv_bias, qk_scale, attn_drop)
        # self.dim = dim
        # self.num_heads = num_heads
        # self.mlp_ratio = mlp_ratio
        # self.norm1 = norm_layer(dim)
        # self.norm1_1 = norm_layer(375)
        self.attn = CrossNeighborhoodAttention(input_size, dim, num_heads,window_size,qkv_bias, qk_scale, attn_drop, drop)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
    # def forward(self, x ,y):
    #     x = x.view(6,-1,8,32)
    #     y = y.view(6,-1,8,32)
    #     shortcut = x
    #     x = self.norm1(x)
    #     y = self.norm1_1(y)
    #     x = self.attn(x,y)
    #     x = shortcut + self.drop_path(x).cuda()
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))

    #     return x
    
    def forward(self, x ,y):
        x = x.view(6,-1,8,32)
        y = y.view(6,-1,8,32)
        shortcut ,tgt2 = self.attn(x,y)
        tgt = shortcut + self.drop_path(tgt2).cuda()
        tgt = self.norm1(tgt)
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))
        tgt = self.drop_path(tgt)
        
        return tgt 
        
def test():
    print('it is cpu')
    model = NATLayer((28,28),128,4)
    img = torch.rand(2,128,56,56)
    try:
        print(model(img).shape)
    except:
        print('error')
        model.set_input_size((56,56))
        print(model(img).shape)
    print('cpu_success\n')

def test_cuda():
    print('it is cuda')
    model = NATLayer((28,28),128,4).cuda()
    img = torch.rand(2,128,56,56).cuda()
    try:
        print(model(img).shape)
    except:
        print('error')
        model.set_input_size((56,56))
        print(model(img).shape)
    print('success')
    print('cuda_success\n')
        
if __name__ == '__main__' :
    test()
    if torch.cuda.is_available():
        test_cuda()