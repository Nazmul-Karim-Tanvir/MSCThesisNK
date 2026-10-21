import torch
from torch import nn

class VideoQA(nn.Module):
    def __init__(self, answers, video_dim=512, text_dim=768, hidden=256, frames=16, question_only=False):
        super().__init__(); self.question_only=question_only
        self.q=nn.Linear(text_dim,hidden)
        self.v=nn.Linear(video_dim,hidden)
        self.pos=nn.Parameter(torch.randn(1,frames,hidden)*.02)
        layer=nn.TransformerEncoderLayer(hidden,4,hidden*4,.1,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.cross=nn.MultiheadAttention(hidden,4,dropout=.1,batch_first=True)
        self.classifier=nn.Sequential(nn.LayerNorm(hidden if question_only else hidden*2),
                                      nn.Linear(hidden if question_only else hidden*2,hidden),nn.GELU(),
                                      nn.Dropout(.1),nn.Linear(hidden,answers))
    def forward(self,video,question):
        q=self.q(question)
        if self.question_only: return self.classifier(q)
        if video.shape[1]!=self.pos.shape[1]: raise ValueError('Frame count differs from checkpoint')
        h=self.temporal(self.v(video)+self.pos)
        c,_=self.cross(q[:,None],h,h,need_weights=False)
        return self.classifier(torch.cat([q,c[:,0]],dim=-1))
