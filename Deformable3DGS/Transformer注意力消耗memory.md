$$Attention(Q,K,V) = softmax(\frac{Q\times K^T}{\sqrt{d_k}})\times V$$
对于$N = batch size, L = squence length$,$Q,K,V$矩阵的size都是
$$[N,n_{heads},L,d_k]$$
$Q*K^T$的结果的形状是
$$[N,n_{heads}, L, L]$$
故消耗的