**StyleMIDI**

风格化 AI 作曲引擎 — 需求与技术路线文档

*从零实现 Transformer · 训练 · 推理 · Web Demo*

v1.0  ·  2026


# **1. 项目概述**
StyleMIDI 是一个从零实现的风格化音乐生成系统，基于 Transformer 架构，能够学习指定作曲家（如贝多芬、肖邦）的创作风格，并根据结构化文本条件（作曲家、情绪、速度、调性）自动生成对应风格的 MIDI 音乐。

## **1.1 核心价值主张**

|**维度**|**内容**|
| :- | :- |
|技术深度|从零实现 Transformer（含 Multi-Head Attention、RoPE、KV Cache）|
|创新点 1|REMI 编码 + 多条件 Token 控制风格，无需预训练文本编码器|
|创新点 2|自动化数据增强：通过乐理算法自动从只含元数据的 MAESTRO MIDI 数据挖掘情绪(Mood)、调性(Key)与速度(Tempo)标签|
|演示效果|Web Demo 一键生成，页面播放音频 + 钢琴卷帘动画|
|算力友好|模型 ~25M 参数，RTX 3060 训练，CPU 即可完成推理|
|延伸实验|注意力可视化、风格插值（贝多芬×肖邦混合），有话题性|



# **2. 需求定义**
## **2.1 功能需求**
### **2.1.1 数据处理模块**
- 下载并解析 MAESTRO Dataset（钢琴 MIDI，按作曲家标注）
- **自动化特征提取与数据均衡 (Data Mining / Augmentation / Balancing)**：利用 pretty_midi 的音级分布 (Pitch Class Profile) 算法推断曲目调性 (Key)；计算音符密度、力度均值等统计学特征并依据 MAESTRO 内部经验分布中位数 (Median) 进行平衡划分，推断情绪 (Mood) 以解决标签不均衡；读取首个速度标记并借助百分位数切分推断相对速度 (Tempo)。
- 使用 REMI 编码将 MIDI 转为 token 序列，词表约 350~400 个 token
- 在每首曲子序列头部注入条件 token：[COMPOSER:xxx]、[MOOD:xxx]、[TEMPO:xxx]、[KEY:xxx]
- 滑动窗口切片，生成固定长度（512 或 1024）的训练样本
- 支持从 MP3 通过 Basic Pitch 转换为 MIDI 再加入训练集

### **2.1.2 模型模块（从零实现）**
- Tokenizer：离散 MIDI 事件词表，支持编码/解码
- Transformer Decoder：N 层 Causal Self-Attention + FFN
- 位置编码：RoPE（Rotary Position Embedding）
- Multi-Head Attention：支持 KV Cache 加速推理
- 条件注入：条件 token 作为前缀，模型学习条件-音乐对应关系

### **2.1.3 训练模块**
- 损失函数：Cross-Entropy（下一个 token 预测）
- 优化器：AdamW + Cosine LR Schedule + Warmup
- 混合精度训练（fp16）节省显存
- Checkpoint 保存与恢复
- 训练日志：loss 曲线实时记录，可用 TensorBoard 查看

### **2.1.4 推理模块**
- 支持 Top-p / Temperature 采样
- CPU 推理：生成 ~512 token（约 30 秒音乐）约 10~30 秒
- token 序列 → REMI 解码 → pretty\_midi 对象 → 输出 .mid 文件
- 可选：FluidSynth 渲染 .mid 为 .mp3

### **2.1.5 Web Demo 模块**
- 框架：Gradio（部署简单，支持 Hugging Face Spaces 一键上线）
- 用户界面：下拉选择作曲家、调性；提供宏观的“情绪”预设选项（如欢快、忧郁），在底层映射为具体的「力度、密度、速度」滑动条组合，用户也可以直接微调 `0.0~1.0` 的连续值滑动条，点击生成
- 结果展示：音频播放器 + 钢琴卷帘可视化（HTML Canvas 或 html-midi-player）
- 可下载生成的 .mid 文件

### **2.1.6 可视化/实验模块（加分项）**
- 注意力热图可视化：展示模型在生成某音符时关注的历史位置
- 风格插值实验：在两个作曲家的条件 embedding 之间插值，生成混合风格音乐
- Temperature 对比：同一条件下不同采样温度的生成结果对比

## **2.2 非功能需求**
- 训练环境：单张消费级 GPU（RTX 3060 / 4060，8GB 显存），单卡即可
- 推理环境：仅需 CPU，无 GPU 依赖，方便任何机器演示
- 代码质量：模块化结构，每个核心组件有详细注释，适合展示和讲解
- 文档质量：README 包含架构图、训练曲线截图、生成样本试听链接
- 可复现性：提供完整的 requirements.txt 和一键训练脚本



# **3. 技术架构**
## **3.1 系统总览**
整个系统分为四个层次，数据流从左到右单向流动：

|**层次**|**模块**|**核心技术/库**|
| :- | :- | :- |
|数据层|MIDI 解析 → REMI 编码 → Dataset|pretty\_midi, miditok, torch.utils.data|
|模型层|Transformer Decoder（从零实现）|PyTorch（纯手写，不用 nn.Transformer）|
|推理层|条件采样 → MIDI 生成|PyTorch CPU, pretty\_midi, FluidSynth|
|展示层|Web Demo + 可视化|Gradio, html-midi-player, matplotlib|

## **3.2 REMI 编码方案**
REMI（Revamped MIDI）将连续的 MIDI 事件离散化为 token 序列，是本项目的核心数据表示。

Token 类型一共五类：

- BAR：小节线，标记每一小节的开始
- POSITION：小节内位置（0~47，将一个小节等分 48 份）
- PITCH：音高（0~127，对应 MIDI 音高编号）
- DURATION：音符时值（16 种，从 1/32 音符到全音符）
- VELOCITY：力度（0~31，量化为 32 档）

条件 Token（前缀注入，每首曲子开头附加）：

- [COMPOSER:beethoven / chopin / mozart / schubert]
- [VELOCITY:0.0~1.0] (力度，连续值)
- [DENSITY:0.0~1.0] (音符密度，连续值)
- [TEMPO:0.0~1.0] (速度，连续值)
- [KEY:C_major / A_minor / ...]（常用 12 个调）

最终词表大小约 380~420 个 token，远小于 NLP 模型的 50000+，模型收敛极快。

## **3.3 模型架构**
### **3.3.1 整体结构**
标准 Transformer Decoder（Causal，仅使用 Masked Self-Attention，无 Cross-Attention）：

|**超参数**|**推荐值**|**说明**|
| :- | :- | :- |
|n\_layers|6|Decoder Block 层数|
|n\_heads|8|注意力头数|
|d\_model|512|隐层维度|
|d\_ff|2048|FFN 中间层维度（4×d\_model）|
|max\_seq\_len|1024|最大序列长度|
|vocab\_size|~400|REMI 词表 + 条件 token|
|dropout|0\.1|训练时 dropout|
|参数量|~25M|RTX 3060 可完整训练|

### **3.3.2 关键实现细节**
- 位置编码使用 RoPE（Rotary Position Embedding），相比 Sinusoidal 对相对位置感知更好，且在推理时支持超过训练长度的序列
- 推理时启用 KV Cache：将已计算的 Key/Value 缓存复用，逐 token 生成时避免重复计算，CPU 上速度提升约 5~10 倍
- Attention 使用 causal mask，确保每个位置只能看到它之前的 token（自回归生成）
- FFN 使用 SwiGLU 激活函数（相比 ReLU 性能更好，与 LLaMA 架构一致，有技术亮点）

## **3.4 训练方案**

|**项目**|**配置**|
| :- | :- |
|数据集|MAESTRO v3.0（200小时钢琴 MIDI，约 4 位作曲家为主）|
|训练样本数|约 50,000~100,000 条（滑动窗口切片后）|
|Batch Size|16~32（根据显存调整）|
|学习率|3e-4，Cosine Decay + 2000 步 Warmup|
|训练步数|约 50,000~100,000 步（视 loss 收敛情况）|
|精度|fp16 混合精度（节省约 40% 显存）|
|预计时间|RTX 3060 约 8~15 小时可完成基础训练|
|Checkpoint|每 5000 步保存一次|

## **3.5 推理方案**
推理完全在 CPU 上运行，流程如下：

1. 构建条件前缀：根据用户选择拼接条件 token，如 [COMPOSER:beethoven][MOOD:energetic][TEMPO:allegro]
1. 自回归生成：模型逐 token 预测，启用 KV Cache，Top-p=0.9，Temperature 可调（0.8~1.2）
1. 终止条件：生成到 512 token 或遇到 [EOS] token
1. REMI 解码：token 序列 → MIDI 事件列表 → pretty\_midi 对象
1. 输出文件：保存 .mid 文件；可选调用 FluidSynth 渲染为 .mp3



# **4. 代码结构**

|**路径**|**内容**|
| :- | :- |
|data/|原始 MIDI 文件存放目录|
|src/tokenizer.py|REMI 编码/解码器，MIDI ↔ token 序列|
|src/dataset.py|PyTorch Dataset，负责加载和切片 token 序列|
|src/model.py|Transformer 完整实现（Attention、RoPE、FFN、KV Cache）|
|src/train.py|训练主循环，含 loss 记录、checkpoint 保存|
|src/generate.py|推理脚本，条件采样 + MIDI 输出|
|src/visualize.py|注意力热图、风格插值实验可视化|
|scripts/extract_features.py|自动在 MAESTRO 数据集上做数据挖掘（调性估算、情绪与速度特征提取）脚本|
|app/demo.py|Gradio Web Demo 入口|
|scripts/convert\_mp3.py|MP3 → MIDI 转换脚本（调用 Basic Pitch）|
|notebooks/experiments.ipynb|Temperature 对比、风格插值等实验记录|
|checkpoints/|训练 checkpoint 存放目录|
|samples/|生成样本 .mid 和 .mp3 文件|
|README.md|项目介绍、训练曲线图、样本试听链接|


# **5. 技术栈**

|**类别**|**库/工具**|**用途**|
| :- | :- | :- |
|深度学习|PyTorch|模型实现与训练（核心）|
|MIDI 处理|pretty\_midi|MIDI 文件读写、音符操作|
|MIDI 编码|miditok|REMI 编码参考实现（可自己重写）|
|音频转换|Basic Pitch (Spotify)|MP3 → MIDI 转换|
|音频渲染|FluidSynth|MIDI → MP3 渲染，用于演示|
|Web Demo|Gradio|交互界面，支持一键部署到 HF Spaces|
|可视化|matplotlib / html-midi-player|注意力热图 + 钢琴卷帘动画|
|实验记录|TensorBoard|训练 loss 曲线实时监控|
|环境管理|conda / pip|requirements.txt 保证可复现|



# **6. 延伸实验**
## **6.1 注意力可视化**
在生成某个音符时，提取各注意力头的权重矩阵，用热图展示模型"在回看哪些历史音符"。预期会发现模型学到了和声关系（同音重复、和弦进行）和节奏模式，这个图放在 README 里视觉冲击力极强。

## **6.2 风格插值实验**
在推理时，将两个作曲家的条件 embedding 按比例加权混合（α×beethoven + (1-α)×chopin），生成一系列插值音乐。通过渐变的 α 值，能听到音乐风格从一个作曲家"过渡"到另一个，是非常有趣且独特的演示内容。

## **6.3 Temperature 对比**
固定相同条件，用 Temperature=0.6 / 0.9 / 1.2 分别生成，展示采样温度对"创造力 vs 保守性"的影响。可以配合钢琴卷帘可视化音符密度，做成对比图放进 README。



# **7. GitHub README 结构建议**
- 项目 Banner（项目名 + 一句话描述 + Demo 链接徽章）
- 在线 Demo 链接（Hugging Face Spaces）+ 效果 GIF 动图
- 生成样本试听区：3~5 段不同风格的 .mp3 嵌入（GitHub 支持音频预览）
- 架构图：从 MIDI 输入到 token 到 Transformer 到输出的流程图
- 注意力热图截图
- 训练 loss 曲线截图
- Quick Start（5 行命令跑起来）
- 技术细节说明（REMI 编码、RoPE、KV Cache 各一段）
- 延伸实验结果（风格插值对比、Temperature 对比）



*StyleMIDI  ·  Build LLM from Scratch  ·  v1.0*