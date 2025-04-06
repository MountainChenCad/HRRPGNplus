import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_, zeros_, kaiming_normal_
from config import Config


class GraphAttention(nn.Module):
    """Multi-head graph attention layer"""

    def __init__(self, in_features, out_features=None, heads=4, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.heads = heads
        self.dropout = dropout
        self.head_dim = self.out_features // heads

        # Multi-head attention projection matrices
        self.W_q = nn.Parameter(torch.FloatTensor(heads, in_features, self.head_dim))
        self.W_k = nn.Parameter(torch.FloatTensor(heads, in_features, self.head_dim))
        self.W_v = nn.Parameter(torch.FloatTensor(heads, in_features, self.head_dim))
        self.W_o = nn.Parameter(torch.FloatTensor(self.out_features, self.out_features))

        # Adjacency matrix projection
        self.W_a = nn.Parameter(torch.FloatTensor(self.out_features, 1))

        # Parameter initialization
        xavier_uniform_(self.W_q)
        xavier_uniform_(self.W_k)
        xavier_uniform_(self.W_v)
        xavier_uniform_(self.W_o)
        xavier_uniform_(self.W_a)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input: x [batch_size, channels, seq_len]
        Output: attention_output [batch_size, out_features, seq_len],
                adj_matrix [batch_size, seq_len, seq_len]
        """
        batch_size, C, N = x.size()

        # Normalize input for stability
        x_norm = F.normalize(x, p=2, dim=1)
        x_t = x_norm.transpose(1, 2)  # [batch_size, seq_len, channels]

        # Multi-head attention
        q_heads = []
        k_heads = []
        v_heads = []

        for head in range(self.heads):
            q = torch.matmul(x_t, self.W_q[head])  # [batch_size, seq_len, head_dim]
            k = torch.matmul(x_t, self.W_k[head])  # [batch_size, seq_len, head_dim]
            v = torch.matmul(x_t, self.W_v[head])  # [batch_size, seq_len, head_dim]

            q_heads.append(q)
            k_heads.append(k)
            v_heads.append(v)

        # Calculate attention scores
        attn_scores = []
        attn_outputs = []

        for q, k, v in zip(q_heads, k_heads, v_heads):
            attn_score = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
            attn_score = F.softmax(attn_score, dim=-1)
            attn_score = self.dropout_layer(attn_score)
            attn_output = torch.bmm(attn_score, v)

            attn_scores.append(attn_score)
            attn_outputs.append(attn_output)

        # Combine multi-head attention results
        attn_output = torch.cat(attn_outputs, dim=-1)  # [batch_size, seq_len, out_features]
        attn_output = torch.matmul(attn_output, self.W_o)  # [batch_size, seq_len, out_features]

        # Generate adjacency matrix
        adj_matrix = torch.mean(torch.stack(attn_scores), dim=0)  # [batch_size, seq_len, seq_len]

        return attn_output.transpose(1, 2), adj_matrix


class FeatureExtractor(nn.Module):
    """Feature extraction module for HRRP data"""

    def __init__(self, in_channels=1, out_channels=1, hidden_dim=16):
        super(FeatureExtractor, self).__init__()

        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),

            # Second conv block
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(),

            # Final conv to restore channel dimension
            nn.Conv1d(hidden_dim * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

        # Initialize with Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        """
        Input: x [batch_size, in_channels, seq_len]
        Output: x [batch_size, out_channels, seq_len]
        """
        return self.conv_layers(x)


class DynamicGraphGenerator(nn.Module):
    """Dynamic graph generation module"""

    def __init__(self, in_channels=1, hidden_channels=64, heads=4, dropout=0.1):
        super(DynamicGraphGenerator, self).__init__()

        # Feature enhancement layer
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU()
        )

        self.attention = GraphAttention(
            in_features=hidden_channels,
            out_features=hidden_channels,
            heads=heads,
            dropout=dropout
        )

        self.lambda_mix = Config.lambda_mix
        self.use_dynamic_graph = Config.use_dynamic_graph

        # Initialize parameters
        for m in self.feature_enhance.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x, static_adj=None):
        """
        Input: x [batch_size, in_channels, seq_len]
        Output: x_enhanced [batch_size, hidden_channels, seq_len],
               dynamic_adj [batch_size, seq_len, seq_len]
        """
        batch_size, _, seq_len = x.size()

        # Feature enhancement
        x_enhanced = self.feature_enhance(x)

        # Attention-driven adjacency matrix generation
        _, attn_adj = self.attention(x_enhanced)

        if not Config.use_dynamic_graph:
            # If not using dynamic graph, return static adjacency matrix
            return x_enhanced, static_adj

        if static_adj is not None:
            # Mix static and dynamic adjacency matrices with random perturbation
            dynamic_lambda = max(0.1, min(0.9, self.lambda_mix))
            dynamic_adj = dynamic_lambda * static_adj + (1 - dynamic_lambda) * attn_adj
        else:
            dynamic_adj = attn_adj

        # Normalize adjacency matrix
        D = torch.sum(dynamic_adj, dim=2, keepdim=True)
        D_sqrt_inv = torch.pow(D, -0.5)
        D_sqrt_inv[torch.isinf(D_sqrt_inv)] = 0.0
        normalized_adj = D_sqrt_inv * dynamic_adj * D_sqrt_inv.transpose(1, 2)

        return x_enhanced, normalized_adj


class GraphConvLayer(nn.Module):
    """Graph convolution layer"""

    def __init__(self, in_channels, out_channels, heads=4, bias=True):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Add batch normalization
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU()

        # Initialize parameters
        orthogonal_(self.weight)
        if self.bias is not None:
            zeros_(self.bias)

    def forward(self, x, adj):
        """
        Input: x [batch_size, in_channels, seq_len]
              adj [batch_size, seq_len, seq_len]
        Output: output [batch_size, out_channels, seq_len]
        """
        x_t = x.transpose(1, 2)  # [batch_size, seq_len, in_channels]

        # Graph convolution operation
        support = torch.matmul(x_t, self.weight)  # [batch_size, seq_len, out_channels]
        output = torch.bmm(adj, support)  # [batch_size, seq_len, out_channels]

        if self.bias is not None:
            output = output + self.bias

        # Apply batch normalization and activation
        output = output.transpose(1, 2)  # [batch_size, out_channels, seq_len]
        output = self.bn(output)
        output = self.activation(output)

        return output


class GlobalAttentionPooling(nn.Module):
    """Global attention pooling layer"""

    def __init__(self, in_channels):
        super(GlobalAttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels // 2, 1)
        )

        # Initialize parameters
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        """
        Input: x [batch_size, in_channels, seq_len]
        Output: pooled [batch_size, in_channels]
        """
        x_t = x.transpose(1, 2)  # [batch_size, seq_len, in_channels]
        attn_score = self.attention(x_t)  # [batch_size, seq_len, 1]
        attn_weight = F.softmax(attn_score, dim=1)  # [batch_size, seq_len, 1]

        # Weighted sum
        weighted_x = x_t * attn_weight  # [batch_size, seq_len, in_channels]
        pooled = torch.sum(weighted_x, dim=1)  # [batch_size, in_channels]

        return pooled


class HRRPGraphNet(nn.Module):
    """HRRP Graph Network with dynamic graph integration"""

    def __init__(self, num_classes, feature_size=None):
        super(HRRPGraphNet, self).__init__()

        # Initialize parameters
        feature_size = feature_size or Config.feature_size
        hidden_channels = Config.hidden_channels
        attention_heads = Config.attention_heads
        dropout = Config.dropout

        # Feature extraction module
        self.feature_extractor = FeatureExtractor(
            in_channels=1,
            out_channels=1
        )

        # Dynamic graph generation module
        self.graph_generator = DynamicGraphGenerator(
            in_channels=1,
            hidden_channels=hidden_channels,
            heads=attention_heads,
            dropout=dropout
        )

        # Graph convolution layers
        self.graph_convs = nn.ModuleList()
        self.graph_convs.append(GraphConvLayer(hidden_channels, hidden_channels))

        for _ in range(Config.graph_conv_layers - 1):
            self.graph_convs.append(GraphConvLayer(hidden_channels, hidden_channels))

        # Pooling layer
        self.pooling = GlobalAttentionPooling(hidden_channels)

        # Classification layer
        self.fc = nn.Linear(hidden_channels, num_classes)
        orthogonal_(self.fc.weight)
        zeros_(self.fc.bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, static_adj=None, return_features=False):
        """
        Input: x [batch_size, 1, seq_len]
              static_adj [batch_size, seq_len, seq_len] optional
        Output: logits [batch_size, num_classes]
        """
        # Apply feature extraction
        x = self.feature_extractor(x)

        # Generate dynamic graph
        x_enhanced, dynamic_adj = self.graph_generator(x, static_adj)

        # Graph convolution
        h = x_enhanced
        for conv in self.graph_convs:
            h = conv(h, dynamic_adj)
            h = self.dropout(h)

        # Graph pooling
        h_pooled = self.pooling(h)

        # Return features if requested
        if return_features:
            return self.fc(h_pooled), h_pooled

        # Classification
        logits = self.fc(h_pooled)

        return logits, dynamic_adj


class MDGN(nn.Module):
    """Meta-Dynamic Graph Network"""

    def __init__(self, num_classes=3):
        super(MDGN, self).__init__()
        self.encoder = HRRPGraphNet(num_classes=num_classes)

        # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad = True

        # Print parameter information for debugging
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MDGN initialized: {total_params} parameters ({trainable_params} trainable)")

    def forward(self, x, static_adj=None, return_features=False):
        # Ensure input data and weights are on the same device
        device = next(self.parameters()).device
        x = x.to(device)
        if static_adj is not None:
            static_adj = static_adj.to(device)

        if return_features:
            return self.encoder(x, static_adj, return_features=True)
        return self.encoder(x, static_adj)

    def clone(self):
        """Create a deep copy of the model for MAML inner loop update"""
        device = next(self.parameters()).device
        clone = MDGN(num_classes=self.encoder.fc.out_features)
        clone.load_state_dict(self.state_dict())
        clone = clone.to(device)
        return clone

    def adapt_params(self, loss, lr=0.01):
        """Update parameters based on loss, return updated parameter dictionary"""
        # Ensure all parameters require gradients
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"Warning: Parameter {name} doesn't require gradients")
                param.requires_grad = True

        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True, allow_unused=True)

        updated_params = {}
        for (name, param), grad in zip(self.named_parameters(), grads):
            if grad is None:
                # If gradient is None, use zero gradient
                updated_params[name] = param
                print(f"Parameter {name} has None gradient")
            else:
                updated_params[name] = param - lr * grad

        return updated_params

    def set_params(self, params):
        """Set parameters from parameter dictionary"""
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name].data
