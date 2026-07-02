import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), H, W

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTBottleneck(nn.Module):
    def __init__(self, embed_dim: int = 512, depth: int = 4, num_heads: int = 8, max_spatial: int = 32, drop: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_spatial * max_spatial, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens, H, W = self.patch_embed(x)
        N = H * W
        pe = self.pos_embed if self.pos_embed.shape[1] == N else F.interpolate(
            self.pos_embed.reshape(1, 32, 32, C).permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1).reshape(1, N, C)

        tokens = tokens + pe
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens).transpose(1, 2).reshape(B, C, H, W)

class ViTUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_ch: int = 64, vit_depth: int = 4, vit_heads: int = 8, drop: float = 0.1):
        super().__init__()
        ch = base_ch
        self.enc1 = ConvBlock(in_channels, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.enc4 = ConvBlock(ch * 4, ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ViTBottleneck(embed_dim=ch * 8, depth=vit_depth, num_heads=vit_heads, max_spatial=32, drop=drop)
        self.up4 = nn.ConvTranspose2d(ch * 8, ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(ch * 16, ch * 8)
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)
        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([e4, self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([e3, self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))
        return self.head(d1)