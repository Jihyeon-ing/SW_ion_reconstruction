import torch.nn as nn

class TSMixerBlock(nn.Module):
    def __init__(self, input_length, input_channels, hidden_dim, dropout_rate=0.1):
        super(TSMixerBlock, self).__init__()

        ### Time-mixing: input (batch, input_length, input_channels) -> permute -> (batch, input_channels, input_length)
        self.time_norm = nn.LayerNorm(input_length)
        self.time_mlp = nn.Sequential(
            nn.Linear(input_length, input_length),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_length, input_length),
            nn.Dropout(dropout_rate),
        )

        ### Feature-mixing
        self.feat_norm = nn.LayerNorm(input_channels)
        self.feat_mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_channels),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # x: (batch, input_length, input_channels)
        # Time-mixing
        y = x.permute(0, 2, 1)
        y = self.time_norm(y)
        y = self.time_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y

        # Feature-mixing
        z = self.feat_norm(x)
        z = self.feat_mlp(z)
        x = x + z
        return x

class TSMixer(nn.Module):
    def __init__(self, input_length, input_channels, output_length, hidden_dim, n_blocks=4, dropout_rate=0.1):
        super(TSMixer, self).__init__()

        self.blocks = nn.ModuleList([
            TSMixerBlock(input_length, input_channels, hidden_dim, dropout_rate) for _ in range(n_blocks)
        ])
        
        self.pre_head_norm = nn.LayerNorm(input_channels)
        self.time_proj = nn.Linear(input_length, 1)
        self.feature_head = nn.Linear(input_channels, output_length)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.pre_head_norm(x)
        x = self.feature_head(x) 
        x = x.permute(0, 2, 1) 
        x = self.time_proj(x)  
        out = x.squeeze(-1)
        return out
    
if __name__ == '__main__':
    model = TSMixer(input_length=7, input_channels=7, output_length=3, hidden_dim=128, dropout_rate=0.1)
    print(model)
