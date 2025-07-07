```python
d_xyz, d_scaling, d_opacity, d_rotation = deform.step(gaussians.get_xyz.detach(),time_input + ast_noise)
```
模型的输入是位置+时间编码，输出是$\Delta p$
时间、位置编码是
$$
\gamma(\mathbf{x}) = \left(
\mathbf{x},
\sin(2^0 \pi \mathbf{x}), \cos(2^0 \pi \mathbf{x}),
\sin(2^1 \pi \mathbf{x}), \cos(2^1 \pi \mathbf{x}),
\dots,
\sin(2^{L-1} \pi \mathbf{x}), \cos(2^{L-1} \pi \mathbf{x})
\right)
$$
