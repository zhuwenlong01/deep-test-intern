# SFT Data Generation Pipeline

一个强大的数据生成pipeline，专门用于生成SFT（Supervised Fine-Tuning）训练数据和评估数据。该pipeline支持多种数据格式，集成了Claude、Gemini、OpenAI等主流LLM API，并提供了完整的数据处理和生成流程。

## 功能特点

### 🔧 核心功能
- **多数据源支持**：支持gala、browsecomp等多种数据集格式
- **智能数据分割**：自动按比例分割训练集和测试集，支持随机打乱
- **多API集成**：支持Claude、Gemini、OpenAI等主流LLM API
- **多轮对话生成**：支持最多10轮的对话生成，模拟真实交互场景
- **断点续传**：支持checkpoint机制，可以从断点继续执行
- **质量控制**：内置数据验证和去重功能
- **多格式输出**：支持Alpaca、ShareGPT、ChatGLM等多种训练格式

### 🚀 高级特性
- **并发处理**：支持多线程并发API调用，提高效率
- **智能重试**：API调用失败时自动重试，支持指数回退
- **负载均衡**：多个API key时自动负载均衡
- **配置灵活**：支持YAML配置文件和命令行参数
- **详细日志**：完整的执行日志和统计信息
- **进度追踪**：实时显示处理进度和剩余时间

## 快速开始

### 1. 环境设置

```bash
# 克隆仓库
git clone <repository-url>
cd sft-data-generation-pipeline

# 安装依赖
pip install -r requirements.txt

# 复制环境变量文件
cp .env.example .env

# 编辑.env文件，添加API keys
```

### 2. 配置API Keys

在`.env`文件中配置至少一个API key：

```bash
# 选择其中一个或多个
CLAUDE_API_KEY=your_claude_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 准备数据

```bash
# 创建目录结构
python main.py --setup-directories

# 生成示例数据（用于测试）
python main.py --create-sample-data
```

### 4. 运行pipeline

```bash
# 基本使用
python main.py --dataset-path data/input/sample_data.json --output-dir data/output

# 使用多个数据集
python main.py --dataset-path data/input/gala.json data/input/browsecomp.json --output-dir data/output

# 使用配置文件
python main.py --config configs/sample_config.yaml --dataset-path data/input/dataset.json
```

## 详细使用指南

### 命令行参数

#### 基本参数
```bash
# 数据集路径（必需）
--dataset-path PATH [PATH ...]     # 输入数据集路径

# 数据集类型（可选）
--dataset-type TYPE [TYPE ...]     # 数据集类型：auto, json, jsonl, csv, gala, browsecomp

# 输出目录
--output-dir DIR                    # 输出目录，默认：data/output
```

#### 配置参数
```bash
# 配置文件
--config CONFIG_FILE                # YAML配置文件路径

# API配置
--claude-api-key KEY               # Claude API key
--gemini-api-key KEY               # Gemini API key  
--openai-api-key KEY               # OpenAI API key

# 生成参数
--max-rounds N                     # 最大对话轮数，默认：10
--temperature FLOAT                # 生成温度，默认：0.7
--batch-size N                     # 批处理大小，默认：16

# 数据分割
--train-ratio FLOAT                # 训练集比例，默认：0.8
--random-seed N                    # 随机种子，默认：42
```

#### 高级参数
```bash
# 断点续传
--resume CHECKPOINT_FILE           # 从checkpoint文件恢复
--enable-resume                    # 启用断点保存
--checkpoint-every N               # 每N个样本保存checkpoint

# 实用工具
--create-sample-data               # 创建示例数据
--setup-directories                # 创建目录结构
--validate-config                  # 验证配置
--dry-run                          # 干运行，不生成数据

# 日志选项
--log-level LEVEL                  # 日志级别：DEBUG, INFO, WARNING, ERROR
--quiet                            # 静默模式
--verbose                          # 详细输出
```

### 使用示例

#### 1. 基本使用
```bash
# 处理单个数据集
python main.py --dataset-path data/input/qa_dataset.json --claude-api-key your_key
```

#### 2. 多数据集处理
```bash
# 处理多个数据集
python main.py \
  --dataset-path data/input/gala.json data/input/browsecomp.json \
  --dataset-type gala browsecomp \
  --output-dir data/output \
  --max-rounds 5
```

#### 3. 使用配置文件
```bash
# 创建配置文件
python main.py --create-sample-config

# 使用配置文件运行
python main.py --config configs/sample_config.yaml --dataset-path data/input/dataset.json
```

#### 4. 断点续传
```bash
# 启用断点续传
python main.py \
  --dataset-path data/input/large_dataset.json \
  --enable-resume \
  --checkpoint-every 50

# 从断点恢复
python main.py \
  --resume checkpoints/train_checkpoint.json \
  --dataset-path data/input/large_dataset.json
```

## 数据格式

### 输入数据格式

Pipeline支持多种输入数据格式：

#### 1. 问答格式
```json
[
  {
    "question": "什么是机器学习？",
    "answer": "机器学习是人工智能的一个分支...",
    "category": "AI"
  }
]
```

#### 2. 指令格式
```json
[
  {
    "instruction": "解释深度学习的基本原理",
    "input": "",
    "output": "深度学习是一种机器学习方法..."
  }
]
```

#### 3. 对话格式
```json
[
  {
    "prompt": "请介绍Python编程语言",
    "response": "Python是一种高级编程语言..."
  }
]
```

### 输出数据格式

Pipeline生成多种格式的输出数据：

#### 1. SFT标准格式 (sft_train_data.jsonl)
```json
{
  "conversations": [
    {
      "instruction": "问题内容",
      "output": "生成的回答",
      "input": "",
      "metadata": {
        "round": 1,
        "api_client": "claude",
        "timestamp": 1234567890
      }
    }
  ],
  "original_data": {...},
  "generation_metadata": {...}
}
```

#### 2. Alpaca格式 (alpaca_train_data.json)
```json
[
  {
    "instruction": "问题内容",
    "input": "",
    "output": "生成的回答"
  }
]
```

#### 3. ShareGPT格式 (sharegpt_train_data.json)
```json
[
  {
    "conversations": [
      {"from": "human", "value": "问题内容"},
      {"from": "gpt", "value": "生成的回答"}
    ],
    "id": "sample_id"
  }
]
```

## 配置文件

### 完整配置示例

```yaml
# API配置
api:
  claude_api_key: null
  gemini_api_key: null
  openai_api_key: null
  max_requests_per_minute: 50
  max_concurrent_requests: 5
  max_retries: 3
  retry_delay: 1.0

# 数据配置
data:
  input_data_path: "data/input"
  output_data_path: "data/output"
  train_ratio: 0.8
  test_ratio: 0.2
  random_seed: 42
  supported_datasets: ["gala", "browsecomp", "custom"]
  shuffle_data: true

# 生成配置
generation:
  max_rounds: 10
  temperature: 0.7
  max_tokens: 2048
  model_preferences:
    - "claude-3-sonnet-20240229"
    - "gemini-1.5-pro"
    - "gpt-4"
  min_answer_length: 10
  max_answer_length: 4000
  filter_duplicates: true
  similarity_threshold: 0.85

# Pipeline配置
log_level: "INFO"
log_file: "logs/pipeline.log"
batch_size: 16
num_workers: 4
enable_resume: true
checkpoint_every: 100
```

## 项目结构

```
sft-data-generation-pipeline/
├── main.py                 # 主入口文件
├── config.py              # 配置管理
├── pipeline.py            # 主pipeline逻辑
├── data_splitter.py       # 数据分割模块
├── api_client.py          # API客户端
├── utils.py               # 工具函数
├── requirements.txt       # 依赖文件
├── .env.example          # 环境变量示例
├── README.md             # 说明文档
├── configs/              # 配置文件目录
│   └── sample_config.yaml
├── data/                 # 数据目录
│   ├── input/           # 输入数据
│   └── output/          # 输出数据
├── logs/                # 日志目录
└── checkpoints/         # 断点文件目录
```

## 常见问题

### Q: 如何处理API调用失败？
A: Pipeline内置了重试机制，支持指数回退。如果一个API失败，会自动尝试其他可用的API。

### Q: 如何处理大数据集？
A: 使用`--batch-size`参数调整批处理大小，使用`--enable-resume`启用断点续传功能。

### Q: 如何控制生成质量？
A: 调整`temperature`参数控制生成的创造性，使用`min_answer_length`和`max_answer_length`控制回答长度。

### Q: 如何避免重复数据？
A: Pipeline内置去重功能，可以通过`similarity_threshold`参数调整去重阈值。

### Q: 如何添加新的数据格式支持？
A: 在`data_splitter.py`中添加新的加载函数，并在`_create_prompt_from_sample`中添加对应的prompt生成逻辑。

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 致谢

本项目参考了以下开源项目的设计思想：
- [Sailcraft](https://github.com/sail-sg/sailcraft) - 数据处理工具包
- [Web Dancer/Sailor](https://github.com/sail-sg/sailcraft) - SFT训练方法

## 更新日志

### v1.0.0
- 初始版本发布
- 支持多种数据格式和API
- 实现多轮对话生成
- 添加断点续传功能
- 支持多种输出格式
