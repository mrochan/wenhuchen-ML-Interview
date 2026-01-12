import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Standard RoPE implementation.
    x: [B, T, NH, D]
    cos/sin: [1, T, 1, D]
    """
    # Split into even and odd indices for rotation
    d = x.shape[-1]
    x_left = x[..., : d // 2]
    x_right = x[..., d // 2 :]
    
    # Standard rotation: [x1, x2] -> [-x2, x1]
    rotated_x = torch.cat([-x_right, x_left], dim=-1)
    
    return (x * cos) + (rotated_x * sin)


class MLAAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_c, d_c_prime, d_rope):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_c = d_c                # KV compression dimension
        self.d_c_prime = d_c_prime    # Query compression dimension
        self.d_rope = d_rope          # Dimension for Decoupled RoPE
        
        # 1. KV Compression path
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)  # Equation 9
        self.W_UK = nn.Linear(d_c, n_heads * d_head, bias=False) # Equation 10
        self.W_UV = nn.Linear(d_c, n_heads * d_head, bias=False) # Equation 11
        
        # 2. Query Compression path
        self.W_DQ = nn.Linear(d_model, d_c_prime, bias=False)    # Equation 12
        self.W_UQ = nn.Linear(d_c_prime, n_heads * d_head, bias=False) # Equation 13
        
        # 3. Decoupled RoPE path
        self.W_QR = nn.Linear(d_c_prime, n_heads * d_rope, bias=False) # Equation 14
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)             # Equation 15
        
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)    # Equation 19
        self.scaling = (d_head + d_rope) ** -0.5

    def forward(self, h, cos, sin, mask=None, kv_cache=None):
        b, t, _ = h.shape
        
        # --- Step 1: Compress Keys & Values ---
        c_kv = self.W_DKV(h)  # [B, T, d_c]
        
        # --- Step 2: Compress & Decompress Queries ---
        c_q = self.W_DQ(h)    # [B, T, d_c_prime]
        q_content = self.W_UQ(c_q).view(b, t, self.n_heads, self.d_head)
        
        # --- Step 3: Decoupled RoPE ---
        # Query rotation
        q_rope = self.W_QR(c_q).view(b, t, self.n_heads, self.d_rope)
        q_rope = apply_rope(q_rope, cos, sin)
        
        # Key rotation (Shared key across heads)
        k_rope = self.W_KR(h).view(b, t, 1, self.d_rope) 
        k_rope = apply_rope(k_rope, cos, sin)
        
        # --- Step 4: KV Projection (Content) ---
        # During training, we explicitly expand; during inference, we "absorb"
        k_content = self.W_UK(c_kv).view(b, t, self.n_heads, self.d_head)
        v_content = self.W_UV(c_kv).view(b, t, self.n_heads, self.d_head)
        
        # Concatenate Content + RoPE parts
        # Equations 16 & 17: q = [q_c; q_r], k = [k_c; k_r]
        q = torch.cat([q_content, q_rope], dim=-1) # [B, T, NH, d_h + d_r]
        k = torch.cat([k_content, k_rope.expand(-1, -1, self.n_heads, -1)], dim=-1)
        
        # --- Step 5: Attention ---
        q = q.transpose(1, 2) # [B, NH, T, D]
        k = k.transpose(1, 2) # [B, NH, T, D]
        v = v_content.transpose(1, 2) # [B, NH, T, D]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            scores += mask
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v) # [B, NH, T, d_head]
        
        # --- Step 6: Final Projection ---
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.W_O(out) # Equation 19


class MLAInferenceAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_c, d_c_prime, d_rope):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_c = d_c                
        self.d_c_prime = d_c_prime    
        self.d_rope = d_rope          

        # --- Training-style weights (Still needed to define the model) ---
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)  
        self.W_UK = nn.Parameter(torch.randn(n_heads, d_head, d_c)) # Per-head up-projection
        self.W_UV = nn.Parameter(torch.randn(n_heads, d_head, d_c)) 
        
        self.W_DQ = nn.Linear(d_model, d_c_prime, bias=False)    
        self.W_UQ = nn.Parameter(torch.randn(n_heads, d_head, d_c_prime)) 
        
        self.W_QR = nn.Linear(d_c_prime, n_heads * d_rope, bias=False) 
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)             
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)    
        
        self.scaling = (d_head + d_rope) ** -0.5

    def get_absorbed_weights(self):
        """
        Computes the absorbed weights for inference.
        These can be cached so they aren't recomputed every forward pass.
        """
        W_UQ_heads = self.W_UQ # [NH, d_head, d_c_prime]
        W_UK_heads = self.W_UK # [NH, d_head, d_c]
        W_absorbed_QK = torch.einsum('nhq,nhc->qnc', W_UQ_heads, W_UK_heads)

        W_O_heads = self.W_O.weight.view(self.W_O.out_features, self.n_heads, self.d_head)
        W_absorbed_VO = torch.einsum('onh,nhc->onc', W_O_heads, self.W_UV) # [d_model, NH, d_c]
        
        return W_absorbed_QK, W_absorbed_VO

    def forward(self, h, cos, sin, mask=None):
        b, t, _ = h.shape
        W_QK, W_VO = self.get_absorbed_weights()
        
        # --- KV Path ---
        c_kv = self.W_DKV(h)  # [B, T, d_c] - This is the ONLY thing cached for content
        k_rope = apply_rope(self.W_KR(h).view(b, t, 1, self.d_rope), cos, sin) # Position key

        # --- Query Path ---
        c_q = self.W_DQ(h)    # [B, T, d_c_prime]
        q_rope = apply_rope(self.W_QR(c_q).view(b, t, self.n_heads, self.d_rope), cos, sin)

        # --- Step 1: Absorbed QK Attention (Low-rank Dot Product) ---
        # Instead of projecting to d_head, project c_q to d_c space
        # q_absorbed: [B, NH, T, d_c]
        q_absorbed = torch.einsum('btq,qnc->btnc', c_q, W_QK) 
        
        # Dot product with compressed keys: [B, NH, T, S]
        content_scores = torch.einsum('btnc,bsc->bnts', q_absorbed, c_kv)
        
        # Add RoPE scores separately (Decoupled RoPE)
        rope_scores = torch.matmul(q_rope.transpose(1, 2), k_rope.transpose(1, 2).transpose(-2, -1))
        
        attn = F.softmax((content_scores + rope_scores) * self.scaling, dim=-1)

        # --- Step 2: Absorbed VO Projection ---
        # First: Weight the latent vectors c_kv by attention
        # latent_out: [B, NH, T, d_c]
        latent_out = torch.matmul(attn, c_kv.unsqueeze(1).expand(-1, self.n_heads, -1, -1))
        
        # Second: Apply the combined W_O * W_UV matrix once at the end
        # final_out: [B, T, d_model]
        out = torch.einsum('bnth,onh->bto', latent_out, W_VO)
        
        return out


class MLATransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_c, d_c_prime, d_rope, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        # self.attn = MLAAttention(d_model, n_heads, d_head, d_c, d_c_prime, d_rope)
        self.attn = MLAInferenceAttention(d_model, n_heads, d_head, d_c, d_c_prime, d_rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_head, d_c, d_c_prime, d_rope, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MLATransformerBlock(d_model, n_heads, d_head, d_c, d_c_prime, d_rope, d_model * 4)
            for _ in range(n_layers)
        ])
        self.max_seq_len = max_seq_len
        self.d_rope = d_rope
        
        # Precompute RoPE sinusoids
        self.register_buffer("cos", self._get_sinusoids(max_seq_len, d_rope, "cos"))
        self.register_buffer("sin", self._get_sinusoids(max_seq_len, d_rope, "sin"))

    def _get_sinusoids(self, length, dim, type="cos"):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(length).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() if type == "cos" else emb.sin()

    def forward(self, x, mask=None):
        b, t = x.shape
        h = self.embed(x)
        
        # Slice RoPE for the current sequence length
        cos = self.cos[:t, :].view(1, t, 1, self.d_rope)
        sin = self.sin[:t, :].view(1, t, 1, self.d_rope)
        
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        return h


if __name__ == '__main__':
    model = Transformer(
        vocab_size=100,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_head=8,
        d_c=4,
        d_c_prime=5,
        d_rope=2,
        max_seq_len=1000
    )

    x = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6]])

    y = model(x)

    print(y)

