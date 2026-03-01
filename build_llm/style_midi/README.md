# StyleMIDI 🎵

*English | [中文(Chinese)](#stylemidi-中文版)*

**Stylized AI Music Composition Engine**
*Building a Transformer from Scratch · Training · Inference · Web Demo*

StyleMIDI is a stylized music generation system built entirely from scratch based on the Transformer architecture. It learns the creative styles of specific composers (e.g., Beethoven, Chopin) and automatically generates MIDI music corresponding to given structured text conditions (Composer, Mood, Tempo, Key).

---

## 🌟 Core Features

- **Built from Scratch**: Fully implements a Transformer Decoder including Multi-Head Attention, RoPE (Rotary Position Embedding), and KV Cache.
- **Innovative Conditioning**: Utilizes REMI encoding combined with multi-conditional Tokens to control musical style, eliminating the need for a pre-trained text encoder.
- **Automated Data Augmentation/Mining**: Extracts Key, Mood, and Tempo labels automatically from MAESTRO MIDI data using music theory algorithms to balance data distribution.
- **Interactive Web UI**: A React-based interface with a FastAPI backend, allowing one-click generation, audio playback, and piano roll visualization.
- **Hardware Friendly**: Lightweight model (~25M parameters) that can be fully trained on a consumer GPU (e.g., RTX 3060 8GB) and inferred on CPU.

## 🏗️ Architecture

The system flows through four main layers:
1. **Data Layer**: MIDI Parsing → REMI Encoding → Dataset (`pretty_midi`, `miditok`)
2. **Model Layer**: Custom Transformer Decoder (PyTorch, without using `nn.Transformer`)
3. **Inference Layer**: Conditional Sampling → MIDI Generation (CPU Inference, KV Cache acceleration)
4. **Presentation Layer**: Web UI + Visualization (React, FastAPI, HTML Canvas / `html-midi-player`)

### Model Specifications
- **Layers**: 6
- **Attention Heads**: 8
- **Hidden Dimension**: 512
- **Max Sequence Length**: 1024
- **Vocabulary Size**: ~400 (REMI Tokens + Condition Tokens)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/StyleMIDI.git
cd StyleMIDI

# Install dependencies
pip install -r requirements.txt
```

### Web Interface & Inference

The project relies on a React frontend and a FastAPI backend for inference:

**1. Start the Backend API**
```bash
# From the project root
python api/server.py
```

**2. Start the Frontend UI**
```bash
cd style_midi_ui
npm install
npm run dev
```
*Open the provided local Vite URL (e.g., http://localhost:5173) in your browser, select a composer, adjust sliders, and generate!*

### Training

To train the model from scratch using the MAESTRO dataset:

```bash
# Download dataset, extract features & prepare tokens
python scripts/prepare_data.py

# Start training (reference: result/style_midi_train.ipynb)
python src/train.py
```

*Note: Training results and reference outputs (like notebooks, charts, and weights) are saved in the `result/` directory.*

---

<br>
<br>

# StyleMIDI (中文版) 🎵

**风格化 AI 作曲引擎**
*从零实现 Transformer · 训练 · 推理 · Web Demo*

StyleMIDI 是一个从零实现的风格化音乐生成系统，基于 Transformer 架构，能够学习指定作曲家（如贝多芬、肖邦）的创作风格，并根据结构化文本条件（作曲家、情绪、速度、调性）自动生成对应风格的 MIDI 音乐。

---

## 🌟 核心特性

- **从零构建**: 纯手工打造 Transformer Decoder，包含 Multi-Head Attention、RoPE (旋转位置编码) 和 KV Cache。
- **创新条件控制**: 采用 REMI 编码 + 多条件 Token 控制风格，无需依赖预训练的 NLP 文本编码器。
- **自动化数据挖掘与增强**: 通过乐理算法自动从 MAESTRO 数据集中挖掘调性(Key)、情绪(Mood)与速度(Tempo)标签，解决数据分布不均衡问题。
- **免显卡推理与交互 UI**: 模型仅 ~25M 参数，支持单卡 RTX 3060 完整训练，并可在纯 CPU 环境下极速推理；提供带音频与钢琴卷帘动画的 React 交互界面与 FastAPI 后端支撑。

## 🏗️ 技术架构

系统分为四个核心层次：
1. **数据层**: MIDI 解析 → REMI 编码 → 封装 Dataset (`pretty_midi`, `miditok`)
2. **模型层**: Transformer Decoder 从零实现 (纯手工 PyTorch 实现，不使用 `nn.Transformer`)
3. **推理层**: 条件采样 → MIDI 生成 (支持 CPU 推理，通过 KV Cache 加速 5~10 倍)
4. **展示层**: Web 界面交互与可视化应用 (React, FastAPI, Vite, `html-midi-player`)

### 模型超参数
- **层数 (n_layers)**: 6
- **注意力头数 (n_heads)**: 8
- **隐层维度 (d_model)**: 512
- **最大序列长度**: 1024
- **词表大小**: 约 400 (REMI Token + 条件前缀 Token)

## 🚀 快速开始

### 环境依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/StyleMIDI.git
cd StyleMIDI

# 安装依赖
pip install -r requirements.txt
```

### 运行 Web 界面进行推理

项目展示采用前后端分离架构（前端 React + Vite，后端 FastAPI）：

**1. 启动后端 API 服务**
```bash
# 在项目根目录下执行
python api/server.py
```
*服务默认运行在 http://localhost:8000*

**2. 启动前端 UI**
```bash
cd style_midi_ui
npm install
npm run dev
```
*在浏览器中打开提示的本地地址 (例如 http://localhost:5173)，即可在页面中选择作曲家、调节速度/力度等连续值条件，一键生成风格化曲目！*

### 模型训练

使用 MAESTRO 数据集从头开始训练：

```bash
# 自动化特征提取与数据准备 (含自动下载、特征计算及 token 生成)
python scripts/prepare_data.py

# 启动混合精度训练 (训练参考及日志见 result/style_midi_train.ipynb)
python src/train.py
```
