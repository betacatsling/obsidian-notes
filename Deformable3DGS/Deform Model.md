```python
d_xyz, d_scaling, d_opacity, d_rotation = deform.step(gaussians.get_xyz.detach(),time_input + ast_noise)
```
模型的输入是位置+时间编码，输出是$\Delta p$
