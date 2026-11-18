import numpy as np
DESCRIPTION="One-hot tokens + 5x6 position table; two 1-head attentions, QK and OV each folded to one matrix, softmax over keys, residual adds, readout at position 4."
s="e03f5rf4pfhcfdvgekgj8g97g8ihb1gnlfojg2iftgfpvg5rex0d43fk0ey9f0kczrfvrfkbfnlg44fi9fvkdkggn9dwhf9mf1ifthe8ggmtfegh2ef55gtsgakf4pizjesxjooeepj7l9g5e56feaewof47ebpgwse90frdfkjfzpf1dg75es0eysfrtf66fajfsif8ufinfktfr2fo4fq3gl7fqyfhqgpiec8ftcgj5frifj7gosee7fs5euzf57fb5euffymf9qf5xeu1feze67g55esffhsflofogg2ef2efr9ej6gvfkfvekaed9f3uekvg3hpaafi7fh7e9qdgnfg2nysds0d50g27fmafcfbstfgsfcvf6wdzmhnzgknempedkepge87iau3knfbdf8vf8efgffpjfm1gjyfmqfv9lpziamgpk8d2ccvgrve51hjwh21s1ni9pg44e2cf60f2efmmeqqf01h7wgvzfx7difdjsg2wg5nf3mfgienmg6gffnj54gwpdn0cm2d8c66dm8zgmeh6ogvbezak4yhg5bujcu4hybbwoeh3g5cg4hfcfig1fv7dhae3a56kjptlt7dcodd6"
a=(np.array([int(s[i:i+3],36)for i in range(0,612,3)]).reshape(34,6)-2e4)/1e3
U,B,V,C,W,K=a[:5],a[5:11],a[11:17],a[17:23],a[23:29],a[29:].reshape(6,5)
def model(x,iv):
 n=len(x);r=lambda k,v:iv(k,v.reshape(n,5,-1))
 e=r("em",np.eye(6)[x]+U)
 p=np.exp(e@B@e.swapaxes(1,2));p/=p.sum(-1)[...,None]
 e=r("r0",e+r("a0",p@(e@V)))
 p=np.exp(e@C@e.swapaxes(1,2));p/=p.sum(-1)[...,None]
 return r("r1",e+p@(e@W))[:,4]@K
ALIGN={"em":["embed"],"a0":["attn0.out"],"r0":["resid0.mid"],"r1":["resid1.mid"]}
