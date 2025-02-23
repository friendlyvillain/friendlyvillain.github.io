---
title: Transformer with Pytorch
author: jh
date: 2025-01-08 22:32:52 +0900
categories: [Machine Learning, Transformer]
tags: [ML, Transformer, Encoder, Decoder, Pytorch, Python]
math: true
mermaid: true
comments: true
---

## 1. Introduction

본 포스팅에서는 시계열 데이터 처리에 뛰어난 성능을 보이고 있는 Transfomer 모델을 PyTorch를 사용하여 설계하는 방법을 다룬다. 
Transformer의 핵심인 attention mechanism에 대해 예를 들어서 설명하고, Transformer의 Encoder와 Decoder를 PyTorch로 구현하기 위한 파라미터를 설정하는 방법에 대해 소개한다. 

## 2. Positional Encoding 

Positional Encoding(PE)은 시계열 데이터의 위치 정보를 알려주기 위해 수행한다. 
자연어 처리를 하는 경우 PE를 통해 임베딩된 각각의 단어에 대한 위치 정보를 알려주게 되며 수식은 다음과 같다. 

$$
    PE(pos, 2i) = \text{sin}\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

$$
    PE(pos, 2i+1) = \text{cos}\left(\frac{pos}{10000^{\frac{2i+1}{d_{model}}}}\right)
$$

위의 수식에서 $pos$는 원시 데이터의 위치, $d_{model}$은 임베딩 크기 (feature size), $i$는 임베딩 크기인 $d_{model}$에서의 위치 정보를 의미한다.
PE의 입력으로 (batch_size, sequence_length, d_model) 크기의 입력이 들어오는 경우, 출력 크기 또한 (batch_size, sequence_length_d_model)로 동일하다. 
PE는 위의 수식에 따라 단순히 입력 데이터의 각 sequence 위치에 해당하는 값을 더해주는 역할만 수행하기 때문에 PE를 구성하는 파라미터는 trainable parameter가 아니므로 register_buffer 처리하여 모델 훈련과정에서 가중치가 업데이트 되지 않도록 해야 한다. 

※ 자연어 이외의 데이터 (e.g., 시계열 데이터)에 대해 임베딩을 적용하지 않고, PE의 입력으로 들어오는 경우, 입력 데이터의 크기는 (batch_size, sequence_length, 1) 이지만 PE의 출력은 (batch_size, sequence_length, d_model)이 된다. 

2절에서 설명한 PE를 PyTorch로 구현하면 다음과 같다. 

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # PE output_dim = [batch_size, seq_len, d_model]
        # Summation based positional encoding
        return self.dropout(x + self.pe[:, :x.size(1), :])
```

## 3. Transformer encoder & decoder

전체적인 Transformer 모델의 구조는 PE -> Encoder -> Decoder -> Recover 구조로 이루어진다. 
특히, encoder와 decoder에는 Transformer에서 핵심이 되는 attention mechanism이 적용되며, PyTorch에서 사용하기 위한 간단한 예시를 들면 다음과 같다. 
구체적인 attention mechanism에 대한 원리는 매우 유명한 논문인 [Attention is all you need](#6-reference)을 참고한다.
원시 데이터로 $[word_1, word_2, ..., word_{L}]$인 길이 $L$의 문장이 입력으로 들어오는 경우를 생각하자. 
한편, 원시 데이터에 대해 vectorize를 하기 위해 크기가 $d_e$인 임베딩을 적용한다고 가정하자. 
이 경우, 임베딩이 적용된 원시 데이터를 $E$라 하면, E의 차원은 $L \times d_e$ 가 된다. 
다음으로 임베딩이 적용된 데이터를 attention mechanism에 사용되는 Query, Key, Value로 표현되어야 한다. 
Query, Key, Value를 각각 $Q, K, V$라 하고, attention 크기를 $d_k$라 하면, $Q, K, V$의 크기는 모두 $L \times d_k$ 가 된다. 
$Q, K, V$를 계산하기 위해 다음과 같은 trainable parameter인 $W_Q, W_K, W_V$를 도입하여 표현할 수 있다.

$$
    Q = E W_Q
$$

$$
    K = E W_K
$$

$$
    V = E W_V
$$

위와 같이 구해진 $Q, K, V$에 대해 self-attention을 적용하기 위한 수식은 다음과 같다. 

$$
    Atten(Q, K, V) = softmax\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

Attention이 적용된 결과 또한 Q, K, V의 차원과 동일한 것을 확인할 수 있다. 

한편, 위와 같은 Attention을 $n_h$회 병렬로 수행된 multi-head attention을 적용하는 경우에는, 출력되는 attention 결과는 $L \times (n_h \cdot d_k)$ 가 된다. 
일반적으로 $(n_h \cdot d_k)$ 값을 임베딩 크기와 동일하게 설정하며, 아래와 같이 본 포스팅의 Positional encoding에서 소개한 $d_{model}$ 와 동일한 값이 된다. 

$$
    d_{model} = n_h \cdot d_k
$$

PyTorch의 torch.nn에서 제공하는 클래스를 이용하면 다음과 같이 transformer encoder를 구현할 수 있다. 


```python

from torch.nn import TransformerEncoder, TransformerEncoderLayer

encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,
                                                     dim_feedforward=dim_feedforward,
                                                     batch_first=True, activation='relu')
transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

```

여기서 encoder를 설계할 때, 가장 중요한 매개변수는 d_model과 nhead 이며, $d_model \% nhead == 0$ 을 만족하도록 설정해야 한다. 
본 포스팅에서는 batch_size를 가장 첫번째 차원으로 다루기 위해서 batch_first를 True로 설정하였다.
나머지 매개변수는 설정하지 않으면 자동으로 default 값이 할당된다.  

Transformer decoder 또한 encoder와 동일하게 attention mechanism이 적용되며, PyTorch를 이용한 decoder 구현 방법은 다음과 같이 encoder를 구현 방법과 유사하다. 

```python
decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,
                                                     dim_feedforward=dim_feedforward,
                                                     batch_first=True, activation='relu')
transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
```

## 4. Transformer Encoder+Decoder 통합 구조 구현

본 절에서는 3절에서 설명한 Transformer encoder와 decoder를 통합하여 구현하는 방법을 설명한다. 
PyTorch에서는 Encoder와 Decoder를 분리하지 않고 한 번에 Transformer를 구현할 수 있는 클래스를 제공하지만, Encoder와 Decoder를 분리하여 구현하면 세부적인 튜닝이 가능하기 때문에 본 포스팅에서는 Encoder와 Decoder를 분리해서 구현하는 방법을 다룬다.

```python
class Transformer(nn.Module):
    def __init__(self, feature_size, tgt_output_size, num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward_enc=2048, dim_feedforward_dec=2048, dropout=0.1, pos_encoder_enable=True):
        # feature_size is equivalent to embedding dimension (d_model)
        super().__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.tgt_mask = None
        self.embedding_src = nn.Linear(1, feature_size) 
        self.embedding_tgt = nn.Linear(1, feature_size)
        self.pos_encoder_enable = pos_encoder_enable
        self.pos_encoder = PositionalEncoding(feature_size)
        # Apply n_head multi-head attention
        # d_key, d_query, d_value = d_model // n_head
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout,
                                                     dim_feedforward=dim_feedforward_enc,
                                                     batch_first=True, activation='relu')
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = TransformerDecoderLayer(d_model=feature_size, nhead=8, dropout=dropout,
                                                     dim_feedforward=dim_feedforward_dec,
                                                     batch_first=True, activation='relu')
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(feature_size, tgt_output_size)
        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, tgt):
        src_len = src.shape[1]
        tgt_len = tgt.shape[1]

        if self.src_mask is None or self.src_mask.size(0) != tgt_len:
            device = src.device
            mask = self._generate_square_subsequent_mask(dim1=tgt_len, dim2=src_len).to(device)
            self.src_mask = mask # lower triangular matrix shape with (tgt_len by src_len)

        if self.tgt_mask is None or self.tgt_mask.size(1) != tgt_len:
            device = tgt.device
            mask = self._generate_square_subsequent_mask(dim1=tgt_len, dim2=tgt_len).to(device)
            self.tgt_mask = mask # Lower triangular matrix shape with (tgt_len by tgt_len)

        src = self.embedding_src(src) # Encoder output_dim = [batch_size, src_len, d_model]
        tgt = self.embedding_tgt(tgt) # Encoder output_dim = [batch_size, tgt_len, d_model]

        if self.pos_encoder_enable:
            src = self.pos_encoder(src)  # PE output_dim = [batch_size, src_len, d_model]
            tgt = self.pos_encoder(tgt) # PE output_dim = [batch_size, tgt_len, d_model]

        memory = self.transformer_encoder(src=src) # Encoder output_dim = [batch_size, min(max_len, src_len), d_model]
        output = self.transformer_decoder(tgt=tgt, memory=memory, memory_mask=self.src_mask, tgt_mask=self.tgt_mask) # Decoder output_dim = [batch_size, min(max_len, tgt_len), d_model]
        output = self.generator(output) # Generator output = [batch_size, min(max_len, seq_len), output_size]
        return output

    def _generate_square_subsequent_mask(self, dim1, dim2):
        mask = torch.tril(torch.ones(dim1, dim2) == 1) # Lower Triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask
```

### 4-1. Generator

Transformer의 decoder output 차원은 [batch_szie, tgt_len, d_model] 이고, 아래와 같이, generator layer를 정의하여 Decoder를 통과 시킨 이후, 최종 Transformer 출력 결과를 token에 mapping 시켜 복원한다.

```python
self.generator = nn.Linear(feature_size, tgt_output_size)
```

시계열 데이터를 예측하는 경우에는 generator의 output_size는 1이 된다. 

### 4-2. pos_encoder_enable

자연어를 처리하는 경우에는 PE를 적용하는 경우의 모델의 성능이 높아지지만 데이터의 특성에 따라 PE를 적용하지 않는 편이 더 유리한 경우가 있다. 
이 경우, 임베딩만 적용한 이후 PE를 거치지 않고 바로 Encoder와 Decoder를 통과시켜 사용할 수 있도록 pos_encoder_enable을 Class 변수로 처리하였다. 

### 4-2. _generate_square_subsequent_mask

_generate_square_subsequent_mask는 모델은 학습하는 과정에서 Ground Truth를 아는 상황에서 모델이 훈련되는 것을 방지하기 위해 Lower Triangular Matrix로 Masking 처리하는 역할을 수행한다. 

## 5. Conclusion

본 포스팅에서는 PyTorch를 이용하여 Transformer를 구현하기 위해, Encodder와 Decoder로 각각 분리한 이후, 통합하여 구현하는 방법에 대해 다루었다.
다음 포스팅에서는 본 포스팅에서 다룬 내용을 기반으로 Transformer를 이용하여 시계열 데이터를 예측하는 예제에 대해 다룬다.

## 6. Reference
[Attention is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)