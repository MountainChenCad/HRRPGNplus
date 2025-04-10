import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_, zeros_, kaiming_normal_
from config import Config
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np


class Conv64F(nn.Module):
    """Standard Conv64F backbone for few-shot learning"""

    def __init__(self, in_channels=1):
        super(Conv64F, self).__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.encoder(x).squeeze(-1)  # Remove spatial dimension


class MAMLModel(nn.Module):
    """MAML (Model-Agnostic Meta-Learning) model with Conv64F backbone"""

    def __init__(self, num_classes=3):
        super(MAMLModel, self).__init__()
        self.feature_extractor = Conv64F(in_channels=1)
        self.classifier = nn.Linear(64, num_classes)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, static_adj=None, return_features=False):
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits, None

    def clone(self):
        """Create a deep copy of the model for inner loop update"""
        clone = MAMLModel(num_classes=self.classifier.out_features)
        clone.load_state_dict(self.state_dict())
        return clone

    def adapt_params(self, loss, lr=0.01):
        """Update parameters based on loss, return updated parameter dictionary"""
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)

        updated_params = {}
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - lr * grad

        return updated_params

    def set_params(self, params):
        """Set parameters from parameter dictionary"""
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name].data


class MAMLPlusPlusModel(nn.Module):
    """MAML++ (Improved MAML) model with Conv64F backbone"""

    def __init__(self, num_classes=3):
        super(MAMLPlusPlusModel, self).__init__()
        self.feature_extractor = Conv64F(in_channels=1)
        self.classifier = nn.Linear(64, num_classes)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # MAML++ specific parameters
        self.inner_lrs = nn.ParameterDict({
            'feature_extractor': nn.Parameter(torch.ones(1) * 0.01),
            'classifier': nn.Parameter(torch.ones(1) * 0.01)
        })

    def forward(self, x, static_adj=None, return_features=False):
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits, None

    def clone(self):
        """Create a deep copy of the model for inner loop update"""
        clone = MAMLPlusPlusModel(num_classes=self.classifier.out_features)
        clone.load_state_dict(self.state_dict())
        return clone

    def adapt_params(self, loss, lr=None):
        """Update parameters with per-layer adaptive learning rates"""
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)

        updated_params = {}
        idx = 0

        # Update feature extractor parameters
        for name, param in self.feature_extractor.named_parameters():
            full_name = f'feature_extractor.{name}'
            lr_value = self.inner_lrs['feature_extractor'].item()
            updated_params[full_name] = param - lr_value * grads[idx]
            idx += 1

        # Update classifier parameters
        for name, param in self.classifier.named_parameters():
            full_name = f'classifier.{name}'
            lr_value = self.inner_lrs['classifier'].item()
            updated_params[full_name] = param - lr_value * grads[idx]
            idx += 1

        return updated_params

    def set_params(self, params):
        """Set parameters from parameter dictionary"""
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name].data


class ANILModel(nn.Module):
    """ANIL (Almost No Inner Loop) model with Conv64F backbone"""

    def __init__(self, num_classes=3):
        super(ANILModel, self).__init__()
        self.feature_extractor = Conv64F(in_channels=1)
        self.classifier = nn.Linear(64, num_classes)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, static_adj=None, return_features=False):
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits, None

    def clone(self):
        """Create a deep copy of the model for inner loop update"""
        clone = ANILModel(num_classes=self.classifier.out_features)
        clone.load_state_dict(self.state_dict())
        return clone

    def adapt_params(self, loss, lr=0.01):
        """ANIL only updates classifier parameters"""
        # Get only the classifier parameters
        classifier_params = list(self.classifier.parameters())

        # Compute gradients only for classifier parameters
        grads = torch.autograd.grad(loss, classifier_params, create_graph=True)

        updated_params = {}
        # Copy all parameters first
        for name, param in self.named_parameters():
            updated_params[name] = param.clone()

        # Update only classifier parameters
        for i, (name, param) in enumerate(self.classifier.named_parameters()):
            full_name = f'classifier.{name}'
            updated_params[full_name] = param - lr * grads[i]

        return updated_params

    def set_params(self, params):
        """Set parameters from parameter dictionary"""
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name].data


class MetaSGDModel(nn.Module):
    """Meta-SGD model with Conv64F backbone"""

    def __init__(self, num_classes=3):
        super(MetaSGDModel, self).__init__()
        self.feature_extractor = Conv64F(in_channels=1)
        self.classifier = nn.Linear(64, num_classes)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # Meta-SGD: Learnable per-parameter learning rates
        self.lr_params = nn.ParameterDict()
        for name, param in self.named_parameters():
            self.lr_params[name.replace('.', '_')] = nn.Parameter(torch.ones_like(param) * 0.01)

    def forward(self, x, static_adj=None, return_features=False):
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits, None

    def clone(self):
        """Create a deep copy of the model for inner loop update"""
        clone = MetaSGDModel(num_classes=self.classifier.out_features)
        clone.load_state_dict(self.state_dict())
        return clone

    def adapt_params(self, loss, lr=None):
        """Update parameters with learnable per-parameter learning rates"""
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)

        updated_params = {}
        for (name, param), grad in zip(self.named_parameters(), grads):
            # Get corresponding learning rate parameter
            lr_name = name.replace('.', '_')
            param_lr = self.lr_params[lr_name]

            # Update parameters with per-parameter learning rates
            updated_params[name] = param - param_lr * grad

        return updated_params

    def set_params(self, params):
        """Set parameters from parameter dictionary"""
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name].data

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
            # Mix static and dynamic adjacency matrices with lambda coefficient
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

        # Store attention weights for visualization
        self.last_attention_weights = attn_weight

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

    def forward(self, x, static_adj=None, return_features=False, extract_attention=False):
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

        # Extract attention weights for visualization if requested
        attention_weights = None
        if extract_attention:
            attention_weights = self.pooling.last_attention_weights

        # Return features and attention if requested
        if return_features:
            return self.fc(h_pooled), h_pooled, dynamic_adj, attention_weights

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

    def forward(self, x, static_adj=None, return_features=False, extract_attention=False):
        # Ensure input data and weights are on the same device
        device = next(self.parameters()).device
        x = x.to(device)
        if static_adj is not None:
            static_adj = static_adj.to(device)

        if return_features or extract_attention:
            return self.encoder(x, static_adj, return_features=return_features, extract_attention=extract_attention)
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


# ========== BASELINE MODELS ==========

class CNNModel(nn.Module):
    """1D CNN baseline model for HRRP recognition"""

    def __init__(self, num_classes, feature_size=None):
        super(CNNModel, self).__init__()
        feature_size = feature_size or Config.feature_size

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, static_adj=None, return_features=False, extract_attention=False):
        features = self.cnn(x).squeeze(-1)
        logits = self.fc(features)

        if return_features:
            return logits, features, None, None if extract_attention else None
        return logits, None


class LSTMModel(nn.Module):
    """LSTM baseline model for HRRP recognition"""

    def __init__(self, num_classes, feature_size=None):
        super(LSTMModel, self).__init__()
        feature_size = feature_size or Config.feature_size

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, static_adj=None, return_features=False, extract_attention=False):
        # 重塑用于LSTM的数据：[batch, channels, seq_len] -> [batch, seq_len, channels]
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)

        # 序列长度上的全局平均池化
        features = lstm_out.mean(dim=1)
        logits = self.fc(features)

        if return_features:
            return logits, features, None, None if extract_attention else None
        return logits, None


class GCNModel(nn.Module):
    """Graph Convolutional Network baseline for HRRP recognition"""

    def __init__(self, num_classes, feature_size=None):
        super(GCNModel, self).__init__()
        feature_size = feature_size or Config.feature_size
        hidden_channels = 64

        # Feature extractor
        self.feature_extractor = FeatureExtractor(in_channels=1, out_channels=1)

        # GCN layers without dynamic graph generation
        self.conv1 = GraphConvLayer(1, hidden_channels)
        self.conv2 = GraphConvLayer(hidden_channels, hidden_channels)

        self.pooling = GlobalAttentionPooling(hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

        self.dropout = nn.Dropout(0.3)

    # models.py 中的 GCNModel 类
    def forward(self, x, static_adj=None, return_features=False, extract_attention=False):
        if static_adj is None:
            batch_size, _, seq_len = x.size()
            device = x.device
            # 基于距离创建固定的邻接矩阵
            static_adj = self._create_distance_adj(seq_len).to(device)
            static_adj = static_adj.unsqueeze(0).expand(batch_size, -1, -1)

        # 特征提取
        x = self.feature_extractor(x)

        # GCN层
        x = self.conv1(x, static_adj)
        x = self.dropout(x)
        x = self.conv2(x, static_adj)
        x = self.dropout(x)

        # 池化和分类
        features = self.pooling(x)
        logits = self.fc(features)

        if return_features:
            attention_weights = getattr(self.pooling, 'last_attention_weights', None)
            return logits, features, static_adj, attention_weights if extract_attention else None
        return logits, static_adj

    def _create_distance_adj(self, seq_len):
        """Create adjacency matrix based on distance between nodes"""
        indices = torch.arange(seq_len)
        distances = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))
        # Convert distance to similarity (closer = higher weight)
        adj = 1.0 / (distances + 1)
        return adj


class GATModel(nn.Module):
    """Graph Attention Network baseline for HRRP recognition"""

    def __init__(self, num_classes, feature_size=None):
        super(GATModel, self).__init__()
        feature_size = feature_size or Config.feature_size
        hidden_channels = 64

        # Feature extractor
        self.feature_extractor = FeatureExtractor(in_channels=1, out_channels=1)

        # Feature enhancement layer
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU()
        )

        # GAT layers
        self.gat1 = GraphAttention(hidden_channels, hidden_channels, heads=4)
        self.gat2 = GraphAttention(hidden_channels, hidden_channels, heads=4)

        self.pooling = GlobalAttentionPooling(hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, static_adj=None, return_features=False, extract_attention=False):
        # 特征提取
        x = self.feature_extractor(x)
        x = self.feature_enhance(x)

        # GAT层
        gat1_out, adj1 = self.gat1(x)
        gat1_out = self.dropout(gat1_out)
        gat2_out, adj2 = self.gat2(gat1_out)
        gat2_out = self.dropout(gat2_out)

        # 池化和分类
        features = self.pooling(gat2_out)
        logits = self.fc(features)

        if return_features:
            attention_weights = getattr(self.pooling, 'last_attention_weights', None)
            return logits, features, adj2, attention_weights if extract_attention else None
        return logits, adj2

class PCASVM:
    """PCA + SVM baseline for HRRP recognition"""

    def __init__(self, n_components=None):
        # 不在初始化时创建PCA，而是在fit时动态设置
        self.n_components = n_components
        self.pca = None
        self.svm = SVC(kernel='rbf', probability=True)
        self.is_fitted = False

    def fit(self, X, y):
        """Fit PCA and SVM on training data"""
        # 重塑数据（如果需要）
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        # 动态确定PCA组件数量
        if self.n_components is None:
            # 自动设置组件数，但不超过可用特征/样本数
            max_components = min(X.shape[0], X.shape[1]) - 1
            n_components = min(30, max_components)  # 使用30或可用的最大值
        else:
            # 使用指定的组件数，但确保不超过限制
            n_components = min(self.n_components, min(X.shape[0], X.shape[1]) - 1)

        # 保证组件数至少为1
        n_components = max(1, n_components)

        # 创建并拟合PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)

        # 拟合SVM
        self.svm.fit(X_pca, y)
        self.is_fitted = True

    def predict(self, X):
        """Predict class labels"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        # Reshape if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        # Transform and predict
        X_pca = self.pca.transform(X)
        return self.svm.predict(X_pca)

    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        # Reshape if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        # Transform and predict probabilities
        X_pca = self.pca.transform(X)
        return self.svm.predict_proba(X_pca)


class TemplateMatcher:
    """Template matching baseline for HRRP recognition"""

    def __init__(self, metric='correlation'):
        self.templates = {}
        self.classes = []
        self.metric = metric

    def fit(self, X, y):
        """Create templates for each class"""
        self.classes = np.unique(y)

        # Create average template for each class
        for cls in self.classes:
            # Select samples for this class
            class_samples = X[y == cls]

            # Create average template
            if len(class_samples.shape) > 2:
                # If input is [batch, channels, seq_len]
                template = class_samples.mean(axis=0)
            else:
                # If input is [batch, seq_len]
                template = class_samples.mean(axis=0, keepdims=True)

            self.templates[cls] = template

    def predict(self, X):
        """Predict class by matching against templates"""
        predictions = []

        for sample in X:
            best_score = -float('inf')
            best_class = None

            # Compare with each template
            for cls, template in self.templates.items():
                if self.metric == 'correlation':
                    score = self._correlation(sample, template)
                elif self.metric == 'euclidean':
                    score = -self._euclidean(sample, template)  # Negative so higher is better
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")

                if score > best_score:
                    best_score = score
                    best_class = cls

            predictions.append(best_class)

        return np.array(predictions)

    def _correlation(self, x, template):
        """Compute correlation coefficient"""
        # Flatten if needed
        if len(x.shape) > 1:
            x = x.flatten()
        if len(template.shape) > 1:
            template = template.flatten()

        return np.corrcoef(x, template)[0, 1]

    def _euclidean(self, x, template):
        """Compute Euclidean distance"""
        # Flatten if needed
        if len(x.shape) > 1:
            x = x.flatten()
        if len(template.shape) > 1:
            template = template.flatten()

        return np.sqrt(np.sum((x - template) ** 2))


# ========== ABLATION MODEL VARIANTS ==========

class StaticGraphModel(MDGN):
    """Static graph variant for ablation study"""

    def __init__(self, num_classes=3):
        super(StaticGraphModel, self).__init__(num_classes)
        # Force static graph mode
        self.encoder.graph_generator.use_dynamic_graph = False

    def forward(self, x, static_adj=None, return_features=False):
        # Ensure static graph is used
        return super().forward(x, static_adj, return_features)


class DynamicGraphModel(MDGN):
    """Pure dynamic graph variant for ablation study"""

    def __init__(self, num_classes=3):
        super(DynamicGraphModel, self).__init__(num_classes)
        # Force dynamic graph mode
        self.encoder.graph_generator.use_dynamic_graph = True
        self.encoder.graph_generator.lambda_mix = 0.0  # Pure dynamic (no static component)

    def forward(self, x, static_adj=None, return_features=False):
        # Use only dynamic graph
        return super().forward(x, static_adj=None, return_features=return_features)


class HybridGraphModel(MDGN):
    """Hybrid graph variant with configurable mixing ratio"""

    def __init__(self, num_classes=3, lambda_mix=0.5):
        super(HybridGraphModel, self).__init__(num_classes)
        # Set hybrid graph mode with specified lambda
        self.encoder.graph_generator.use_dynamic_graph = True
        self.encoder.graph_generator.lambda_mix = lambda_mix

    def forward(self, x, static_adj=None, return_features=False):
        return super().forward(x, static_adj, return_features)