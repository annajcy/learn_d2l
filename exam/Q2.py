import torch
import torch.nn as nn
import torch.nn.functional as F

# Grouped Query Attention (GQA) Module
class GroupedQueryAttention(nn.Module):
    """
    num_query_heads: Total number of query heads
    num_kv_heads: Number of key/value heads (must divide num_query_heads)
    The queries are split into groups, each group attending to the same key/value heads.
    
    Args:
        d_model: Dimension of the model
        num_query_heads: Total number of query heads
        num_kv_heads: Number of key/value heads (must divide num_query_heads)
    """
    
    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        assert num_query_heads % num_kv_heads == 0, "num_kv_heads must divide num_query_heads"
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        
        self.num_groups = num_query_heads // num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.kv_dim = self.head_dim * num_kv_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.kv_dim)
        self.W_v = nn.Linear(d_model, self.kv_dim)
        
        self.dense = nn.Linear(d_model, d_model)
        
    def split_query_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_query_heads, seq_len, head_dim)
    
    def split_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_kv_heads, seq_len, head_dim)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, kv_dim)
        V = self.W_v(value)  # (batch_size, seq_len, kv_dim)
        
        Q = self.split_query_heads(Q)  # (batch_size, num_query_heads, seq_len, head_dim)
        K = self.split_kv_heads(K)      # (batch_size, num_kv_heads, seq_len, head_dim)
        V = self.split_kv_heads(V)      # (batch_size, num_kv_heads, seq_len, head_dim)
        
        outputs = []
        for g in range(self.num_groups):
            q_group = Q[:, g * self.num_kv_heads:(g + 1) * self.num_kv_heads, :, :]  # (batch_size, num_kv_heads, seq_len, head_dim)
            attn_output = self.scaled_dot_product_attention(q_group, K, V)  # (batch_size, num_kv_heads, seq_len, head_dim)
            outputs.append(attn_output)
        
        concat_output = torch.cat(outputs, dim=1)  # (batch_size, num_query_heads, seq_len, head_dim)
        concat_output = concat_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_query_heads, head_dim)
        concat_output = concat_output.view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        
        final_output = self.dense(concat_output)  # (batch_size, seq_len, d_model)
        return final_output

if __name__ == "__main__":
    # Example usage:
    # 12 total Q heads, but only 6 heads for K/V => 2 groups of queries
    batch_size, seq_len, d_model = 2, 1024, 768
    num_query_heads = 12
    num_kv_heads = 6 # Must divide num_query_heads

    gqa = GroupedQueryAttention(d_model, num_query_heads, num_kv_heads)
    print("GQA module initialized successfully.")

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    out = gqa(query, key, value)
    print("Output shape:", out.shape)
    assert out.shape == (batch_size, seq_len, d_model)
    print("Test passed!")