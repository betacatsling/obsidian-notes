## Debug
> SAM版本的渲染
## Deform Model的输入与输出
### 1. 位置编码 (Positional Encoding)

项目使用 `utils.time_utils.get_embedder` 进行位置编码。对于任何输入的标量值 x，编码函数 PE(x) 将其映射为高维向量。

- **输入:** 标量 x∈R。
    
- **输出:** 编码向量 PE(x)∈RDembed​。
    
    - 尽管典型的正弦/余弦位置编码输出维度是 2L，但根据您提供的调试信息，`get_embedder` 实际返回的维度有所不同。
        
    - **时间编码维度 (Dtime_emb​):** 对于时间 `fid`，实际观察到 Dtime_emb​=21。
        
    - **空间编码维度 (Dspatial_emb​):** 对于 dx 或 dy 坐标，实际观察到 Dspatial_emb​=12。
        

### 2. `gather_transformer_inputs_batched` 函数的输出 (DeformModel 的输入)

此函数（位于 `train.py`）负责构建 `DeformModel` 的输入特征张量。设批量大小为 Nbatch​，每个中心点有 K 个邻居。

#### a. `center_features` (中心高斯点特征)

- **组成:** 由中心高斯点的三维坐标 (`xyz`) 和时间编码拼接而成。
    
- **来源:**
    
    - `xyz` 坐标 Pxyz(i)​∈R3: 来自 `gaussians.get_xyz[batch_indices]`。
        
    - 时间编码 PE(t)∈RDtime_emb​: 来自 `embed_time_fn(fid)` 的结果。
        
- 数学公式:
    
    Fcenter(i)​=[Pxyz(i)​,PE(t)]
    
    其中 [⋅,⋅] 表示向量拼接。
    
- **维度:** 3+Dtime_emb​=3+21=24。
    
    - 对应的调试输出: `torch.Size([N_batch, 24])`。
        

#### b. `neighbor_features` (邻居高斯点特征)

- **组成:** 由邻居高斯点的几何属性、时间编码、以及其相对于中心点的 dx 和 dy 坐标的位置编码拼接而成。
    
- **来源:**
    
    - **几何属性** G(i,j)∈R7:
        
        - `scale_{xy}^{(i,j)} \in \mathbb{R}^2$: 来自` gaussians.get_scaling_xy`。
            
        - `rot^{(i,j)} \in \mathbb{R}^4`: 来自 `gaussians._rotation` (四元数)。
            
        - `\alpha^{(i,j)} \in \mathbb{R}^1`: 来自 `gaussians.get_opacity`。
            
        - 拼接形成 G(i,j)=[scalexy(i,j)​,rot(i,j),α(i,j)]。
            
    - **时间编码** PE(t)∈RDtime_emb​: 与中心点相同的时间编码。
        
    - **空间相对位置编码** (PE(Δx(i,j)),PE(Δy(i,j))∈RDspatial_emb​):
        
        - Δx(i,j),Δy(i,j) 是邻居点 Ppixel(i,j)​ 与中心点 Ppixel(i)​ 在像素坐标上的直接差值：(Δx(i,j),Δy(i,j))=Ppixel(i,j)​−Ppixel(i)​。
            
        - `spatial_embed_fn` 对 Δx(i,j) 和 Δy(i,j) 分别进行位置编码。
            
- 数学公式:
    
    Fneighbor(i,j)​=[G(i,j),PE(t),PE(Δx(i,j)),PE(Δy(i,j))]
- **维度:** Dgeom​+Dtime_emb​+2⋅Dspatial_emb​=7+21+2⋅12=7+21+24=52。
    
    - 对应的调试输出: `torch.Size([N_batch, K, 52])`。
        

#### c. `relative_positions` (原始相对位置)

- **组成:** 原始的二维像素坐标差值 (Δx,Δy)。
    
- 数学公式:
    
    Rpos(i,j)​=(Δx(i,j),Δy(i,j))
- **维度:** 2 (对于每个邻居)。最终张量形状为 `[N_batch, K, 2]`。
    

### 3. `TransformerDeformNetwork` 的内部嵌入与操作

`TransformerDeformNetwork` (位于 `utils/transformer_utils.py`) 接收上述输入并进行处理。

- **输入特征嵌入 (Input Feature Embedding):**
    
    - 中心点嵌入: 将 Fcenter​ 映射到 Transformer 的模型维度 dmodel​。
        
        Ecenter(i)​=Linearcenter​(Fcenter(i)​)∈Rdmodel​
        - `Linear_center` 的输入维度是 `center_feature_input_dim = 24`。
            
    - 邻居点嵌入: 将 Fneighbor​ 映射到 Transformer 的模型维度 dmodel​。
        
        Eneighbor(i,j)​=Linearneighbor​(Fneighbor(i,j)​)∈Rdmodel​
        - `Linear_neighbor` 的输入维度是 `neighbor_feature_input_dim = 52`。
            
- **相对位置编码 (Position Encoder):**
    
    - 将原始的二维相对位置 Rpos(i,j)​ 映射到 dmodel​ 维，然后加到邻居嵌入上。
        
    - Erel_pos(i,j)​=MLPpos​(Rpos(i,j)​)∈Rdmodel​
        - `MLP_pos` (即 `self.position_encoder`) 的输入维度是 `2` (原始 Δx,Δy)，输出维度是 `d_model`。
            
    - **加法操作:** 邻居嵌入与相对位置编码相加：$\mathbf{E}'_{neighbor}^{(i,j)} = \mathbf{E}_{neighbor}^{(i,j)} + \mathbf{E}_{rel\_pos}^{(i,j)}$。
        
- **Transformer 序列构建:**
    
    - 将中心点嵌入（扩展一维）与处理后的邻居嵌入拼接，形成 Transformer 的输入序列。
        
    - $$\mathbf{S}^{(i)} = [\mathbf{E}_{center}^{(i)}, \mathbf{E}'_{neighbor}^{(i,1)}, \dots, \mathbf{E}'_{neighbor}^{(i,K)}]$$
    - 序列形状: [Nbatch​,1+K,dmodel​]。
        
- **Transformer 编码器:**
    
    - 将序列 S(i) 输入 Transformer 编码器。
        
    - O(i)=TransformerEncoder(S(i))∈R(1+K)×dmodel​
    - `TransformerEncoder` 的输出形状与输入序列形状相同。
        
- **输出预测头 (Output Head):**
    
    - 提取序列的第一个 token（对应中心高斯点经过注意力机制处理后的表示），并输入 MLP 预测最终的变形量。
        
    - D(i)=MLPoutput​(O(i)[0])∈R7
        - `MLP_output` (即 `self.output_head`) 的输入维度是 `d_model`，输出维度是 `7`。
            

### 4. `DeformModel` 的最终输出

`DeformModel` 的 `step` 方法返回由 `output_head` 预测的变形量，这些变形量被分割成不同的部分。

- **二维位移 (dxyz​):** Δpxy(i)​∈R2
    
- **二维缩放变化 (dscaling​):** Δs(i)∈R2
    
- **不透明度变化 (dopacity​):** Δα(i)∈R1
    
- **旋转变化 (drotation​):** Δqrot(i)​∈R2 (注意：虽然四元数通常是 4 维，但您的代码中输出的 `d_rotation` 部分是 2 维，这意味着它可能预测的是旋转的某些特定分量或编码，而不是完整的四元数)。
    

这些变形量最终将应用于高斯点的原始属性，以生成动态帧。