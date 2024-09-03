
## 简介

一个 GPT 的简单实现. 其中后缀 _old 的版本是基于单个 CPU/GPU 的实现, 代码比较简洁, 便于阅读和理解 GPT 的结构; 无后缀 _old 的版本是基于 torch.nn.parallel.DistributedDataParallel 实现的多 GPU 训练代码, 可以进行单机单卡, 单机多卡, 多机多卡的数据并行训练.

## 结构

![gpt-2](https://xiaophai-typora.oss-cn-shanghai.aliyuncs.com/gpt-2.png)

```shell
number of parameters: 85.00M
GPT(
  (transformer): ModuleDict(
    (wte): Embedding(65, 768)
    (wpe): Embedding(256, 768)
    (drop): Dropout(p=0.0, inplace=False)
    (blocks): ModuleList(
      (0-11): 12 x Block(
        (ln_1): LayerNorm()
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=768, out_features=2304, bias=False)
          (c_proj): Linear(in_features=768, out_features=768, bias=False)
          (attn_dropout): Dropout(p=0.0, inplace=False)
          (resid_dropout): Dropout(p=0.0, inplace=False)
        )
        (ln_2): LayerNorm()
        (mlp): MLP(
          (c_fc): Linear(in_features=768, out_features=3072, bias=False)
          (gelu): GELU(approximate='none')
          (c_proj): Linear(in_features=3072, out_features=768, bias=False)
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm()
  )
  (lm_head): Linear(in_features=768, out_features=65, bias=False)
)
```

## PreTrain

使用 [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) 文本, 在 character-level 下训练 50,000 个 iteration. 权重文件已经 push 至 [whut-zhangwx/SimpleGPT | huggingface](https://huggingface.co/whut-zhangwx/SimpleGPT)

## 示例

input:

```shell
"First Citizen:\nBefore we proceed any further, hear me speak."
```

output:

```shell
start  generate
---------------
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on w
---------------
finish generate
```

## 参考

论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

博客

- [深入理解GPT | whut-zhangwx](https://whut-zhangwx.github.io/deep-into-gpt/)
