
### 1. 位置编码 (Positional Encoding)

项目使用 `utils.time_utils.get_embedder` 函数进行位置编码。它将一个标量值 x 映射为一个高维向量。

- **输入:** 标量 x∈R (例如时间 `fid` 或相对坐标 `dx`, `dy`)。
    
- **输出:** 编码向量 PE(x)∈RDembed​。
    
    - **时间编码维度 (Dtime_emb​):** 根据调试输出，实际 observed Dtime_emb​=21。
        
    - **空间编码维度 (Dspatial_emb​):** 根据调试输出，实际 observed Dspatial_emb​=12。
        

### 2. `gather_transformer_inputs_batched` 函数的输出 (DeformModel 的输入)

此函数（位于 `train.py`）负责构建传递给 `DeformModel` 的特征张量。设批量大小为 Nbatch​，每个中心点有 K 个邻居。

#### a. `center_features` (中心高斯点特征)

- **组成:** 中心高斯点的三维坐标 (`xyz`) 和时间编码的拼接。
    
- **来源:**
    
    - `xyz` 坐标 Pxyz(i)​∈R3: 来自 `gaussians.get_xyz`。
        
    - 时间编码 PE(t)∈RDtime_emb​: 来自 `embed_time_fn(fid)`。
        
- 数学公式:
    
    Fcenter(i)​=[Pxyz(i)​,PE(t)]
    
    其中 [⋅,⋅] 表示向量拼接。
    
- **维度:** 3+Dtime_emb​=3+21=24。
    
    - 调试输出示例: `torch.Size([N_batch, 24])`。
        

#### b. `neighbor_features` (邻居高斯点特征)

- **组成:** 根据最新修改，邻居高斯点的特征也仅由其三维坐标 (`xyz`) 和时间编码组成，与中心高斯点特征的内部结构相同。
    
- **来源:**
    
    - `xyz` 坐标 Pxyz(i,j)​∈R3: 来自 `gaussians.get_xyz`。
        
    - 时间编码 PE(t)∈RDtime_emb​: 与中心点相同的时间编码。
        
- 数学公式:
    
    Fneighbor(i,j)​=[Pxyz(i,j)​,PE(t)]
- **维度:** 3+Dtime_emb​=3+21=24。
    
    - 调试输出示例: `torch.Size([N_batch, K, 24])`。
        

#### c. `relative_positions` (原始相对位置)

- **组成:** 邻居高斯点相对于中心高斯点在图像像素坐标系中的原始二维像素坐标差值 (Δx,Δy)。
    
- **来源:** 通过像素坐标相减计算。
    
- 数学公式:
    
    Rpos(i,j)​=(Δx(i,j),Δy(i,j))
- **维度:** `[N_batch, K, 2]`。
    

### 3. ransformer 的模型维度 dmodel​ (例如 256)。
        
        Ecenter(i)​=Linearcenter​(Fcenter(i)​)∈Rdmodel​
        - `Linear_center` (即 `self.`TransformerDeformNetwork` 的内部嵌入与操作

`TransformerDeformNetwork` (位于 `utils/transformer_utils.py`) 接收上述输入并进行处理。

- **输入:**
    
    - `center_features`: Fcenter​∈RNbatch​×24
        
    - `neighbor_features`: Fneighbor​∈RNbatch​×K×24
        
    - `relative_positions`: Rpos​∈RNbatch​×K×2 (用于 `position_encoder`)
        
- **内部嵌入 (Input Embedding Layers):**
    
    - 中心点输入嵌入: 将 center_features 映射到 Tcenter_input_embedding`) 的输入维度是 `24`。
            
    - 邻居点输入嵌入: 将 neighbor_features 映射到 Transformer 的模型维度 dmodel​。
        
        Eneighbor(i,j)​=Linearneighbor​(Fneighbor(i,j)​)∈Rdmodel​
        - `Linear_neighbor` (即 `self.neighbor_input_embedding`) 的输入维度是 `24`。
            
- **相对位置编码 (Position Encoder):**
    
    - 将原始的二维相对位置 Rpos(i,j)​ (形状 `[N_batch, K, 2]`) 映射到 dmodel​ 维，然后加到邻居嵌入上。
        
    - Erel_pos(i,j)​=MLPpos​(Rpos(i,j)​)∈Rdmodel​
        - `MLP_pos` (即 `self.position_encoder`) 的输入维度是 `2` (原始 Δx,Δy)，输出维度是 `d_model`。
            
    - **加法操作:** 邻居嵌入与相对位置编码相加：$\mathbf{E}'_{neighbor}^{(i,j)} = \mathbf{E}_{neighbor}^{(i,j)} + \mathbf{E}_{rel\_pos}^{(i,j)}$。
        
- **Transformer 序列构建:**
    
    - 将中心点嵌入（扩展一维）与处理后的邻居嵌入拼接，形成 Transformer 的输入序列。
        
    - $$\mathbf{S}^{(i)} = [\mathbf{E}_{center}^{(i)}, \mathbf{E}'_{neighbor}^{(i,1)}, \dots, \mathbf{E}'_{neighbor}^{(i,K)}]$$
    - 序列形状: `[N_batch, 1+K, d_model]`。
        
- **Transformer 编码器:**
    
    - 将序列 S(i) 输入 `TransformerEncoder`。
        
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
    
- **旋转变化 (drotation​):** Δqrot(i)​∈R2 (这表示预测的是旋转的某些特定分量，而不是完整的四元数)。
    

这些变形量最终将应用于高斯点的原始属性，以生成动态场景的渲染帧。