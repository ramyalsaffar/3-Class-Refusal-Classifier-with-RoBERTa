# Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection

A production-ready, dual-task classification system using fine-tuned RoBERTa models for comprehensive LLM safety analysis:
1. **Refusal Classification** (3 classes): No Refusal, Hard Refusal, Soft Refusal
2. **Jailbreak Detection** (2 classes): Attack Failed, Attack Succeeded

## 🆕 What's New in V09

**V09** introduces major performance and reliability improvements:

### **🚀 Parallel Processing & Checkpointing (5-10x Speedup)**
- **Parallel API Calls**: ThreadPoolExecutor-based concurrent processing
  - 5 workers locally, 10 workers on AWS
  - Response collection: ~500 responses checkpointed
  - Labeling: ~100 samples checkpointed
- **Automatic Error Recovery**: Resume from last checkpoint after crashes/interruptions
- **Smart Cleanup**: Keeps last 2 checkpoints, auto-deletes old ones

### **🛡️ Smart Validation for Modern LLMs**
- **Zero-Sample Handling**: Gracefully handles cases where modern LLMs (Claude 4, GPT-5, Gemini 2.5) successfully defend against ALL jailbreak attempts
- **Minimum Sample Check**: Validates at least 10 samples per class before training
- **Clear Messaging**: Comprehensive explanation when jailbreak detector is skipped
- **No Crashes**: Weighted loss calculation supports zero-count classes

### **🔄 WildJailbreak Dataset Integration (NEW)**
- **Intelligent Data Supplementation**: Automatically supplements training data from [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) when insufficient real jailbreak samples
- **Threshold-Based**: Targets 50% of total_prompts (e.g., 2000 prompts → 1000 jailbreak samples needed)
- **Diverse Sampling**: Stratified sampling across jailbreak tactics for maximum diversity
- **Full Transparency**:
  - Real-time composition reporting (% real vs. WildJailbreak)
  - Source tracking in all outputs and reports
  - Warning system if >30% supplemented
- **Quality Filters**: Length validation and deduplication
- **Reproducible**: Uses config seed for consistent sampling

**Example Output:**
```
📈 JAILBREAK TRAINING DATA COMPOSITION
Real data:         45 samples (35.2%)
WildJailbreak:     83 samples (64.8%)
Total succeeded:  128 samples
⚠️  WARNING: 64.8% of data from WildJailbreak
```

### **🏗️ Improved Architecture**
- **File Reorganization**: `01-Constants.py` centralizes environment detection, paths, device config
- **Loading Order**: Constants → Config → Modules (proper dependency management)
- **Async Imports**: Pre-loaded threading support for parallel processing

### **Performance Impact**
| Operation | V08 (Sequential) | V09 (Parallel) | Speedup |
|-----------|-----------------|----------------|---------|
| Response Collection (2000 prompts × 3 models) | ~30 minutes | ~5 minutes | **6x faster** |
| LLM Judge Labeling (6000 samples) | ~45 minutes | ~8 minutes | **5.6x faster** |
| **Total Pipeline** | **~75 minutes** | **~13 minutes** | **5.8x faster** |

---

## 🧠 Why RoBERTa?

**RoBERTa** (Robustly Optimized BERT Pretraining Approach) was selected as the backbone model based on extensive literature demonstrating its superiority for text classification tasks, particularly in safety-critical domains:

### **Literature Support:**

1. **Superior Text Classification Performance** (Liu et al., 2019)
   - RoBERTa achieves state-of-the-art results on GLUE, RACE, and SQuAD benchmarks
   - Outperforms BERT through: dynamic masking, larger batch sizes, removal of Next Sentence Prediction (NSP)
   - Trained on 160GB of text (10x more than BERT) with longer sequences

2. **Robust for Safety & Toxicity Detection** (Vidgen et al., 2021; Pozzobon et al., 2023)
   - RoBERTa-based models consistently outperform alternatives on hate speech and toxic content detection
   - Strong performance on nuanced classification tasks requiring contextual understanding

3. **Effective for Refusal Pattern Recognition** (Qi et al., 2023; Röttger et al., 2024)
   - Transformer models like RoBERTa excel at capturing linguistic patterns in LLM safety behaviors
   - Bidirectional attention mechanism critical for understanding subtle refusal cues ("soft refusals")

4. **Production-Ready & Well-Supported**
   - Extensive Hugging Face ecosystem with 12,000+ RoBERTa checkpoints
   - Efficient fine-tuning with minimal compute requirements
   - Proven deployment at scale (OpenAI, Google, Meta)

### **Why Not Baseline Comparisons?**

**No baseline analysis was conducted** as the literature already establishes RoBERTa as the optimal choice for:
- Multi-class text classification
- Contextual understanding of nuanced language
- Safety-critical NLP applications
- Production deployment constraints

**Key References:**
- Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." *arXiv:1907.11692*
- Vidgen, B., et al. (2021). "Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection." *ACL 2021*
- Qi, X., et al. (2023). "Fine-tuning Aligned Language Models Compromises Safety." *ICLR 2024*

---

## 🎯 Features

### **Core Capabilities:**
- ✅ **Dual-Task Classification**: Simultaneously detects refusal patterns AND jailbreak success
- ✅ **Production-Ready**: Full API server with A/B testing, monitoring, and auto-retraining
- ✅ **Comprehensive Interpretability**: SHAP analysis, attention visualization, power law analysis
- ✅ **LLM Judge Labeling**: GPT-4o-based labeling with confidence scores and randomized class ordering
- ✅ **3-Stage Prompt Generation**: Quality-controlled prompt generation pipeline
- ✅ **Multi-Model Data Collection**: Claude Sonnet 4.5, GPT-5, Gemini 2.5 Flash
- ✅ **Generic Design**: Works with any N-class classifier (not hardcoded to 3 classes)

### **Production Features:**
- 🚀 **FastAPI Server**: Real-time inference with A/B testing
- 📊 **Monitoring System**: Automatic drift detection with escalating validation
- 🔄 **Auto-Retraining**: Triggered retraining when performance degrades
- 📄 **PDF Report Generation**: Professional reports using ReportLab
- 🗄️ **PostgreSQL Integration**: Production data management
- ☁️ **AWS Integration**: Secrets Manager support
- 🐳 **Docker & Containerization**: Multi-stage builds with GPU support for easy deployment

### **Research & Evaluation Features (Phase 2):**
- 🔬 **5-Fold Cross-Validation**: Stratified k-fold CV with held-out test set
- 📊 **Statistical Hypothesis Testing**: Chi-square tests for class balance analysis
- 🔍 **Comprehensive Error Analysis**: 7 detailed analysis modules
  - Confusion matrix deep dive
  - Per-class performance breakdown
  - Confidence analysis (correct vs incorrect predictions)
  - Input length analysis (accuracy by token length)
  - Failure case extraction (top 50 most confident mistakes)
  - Token-level attribution (attention-based)
  - Jailbreak-specific error analysis
- 📈 **Publication-Ready Results**: Mean ± std metrics across folds with statistical rigor

---

## 📁 Project Structure

```
Dual-RoBERTa-Classifiers/
├── src/
│   ├── 01-Imports.py               # Central import manager
│   ├── 02-Setup.py                 # Environment, paths, device config, class labels
│   ├── 03-Utils.py                 # Utility functions (KeepAwake, DynamicRateLimiter, formatting)
│   ├── 04-Config.py                # All configuration settings (Control Room)
│   ├── 05-CheckpointManager.py     # Checkpoint management for error recovery
│   ├── 06-AWS.py                   # AWS configuration and Secrets Manager
│   ├── 07-PromptGenerator.py       # 3-stage prompt generation with personas
│   ├── 08-ResponseCollector.py     # Multi-LLM response collection + parallel processing
│   ├── 09-DataCleaner.py           # Comprehensive data cleaning
│   ├── 10-DataLabeler.py           # LLM judge labeling + parallel processing
│   ├── 11-WildJailbreakLoader.py   # WildJailbreak dataset loader for supplementation
│   ├── 12-LabelingQualityAnalyzer.py # Quality analysis for labeled data
│   ├── 13-ClassificationDataset.py # PyTorch Dataset
│   ├── 14-DatasetValidator.py      # Dataset validation
│   ├── 15-RefusalClassifier.py     # 3-class RoBERTa model
│   ├── 16-JailbreakDetector.py     # 2-class RoBERTa model
│   ├── 17-Trainer.py               # Trainer with weighted loss + zero-sample handling
│   ├── 18-CrossValidator.py        # K-fold cross-validation
│   ├── 19-PerModelAnalyzer.py      # Per-model performance analysis
│   ├── 20-ConfidenceAnalyzer.py    # Confidence score analysis
│   ├── 21-AdversarialTester.py     # Paraphrasing robustness tests
│   ├── 22-JailbreakAnalysis.py     # Security-focused jailbreak analysis
│   ├── 23-CorrelationAnalysis.py   # Refusal ↔ Jailbreak correlation
│   ├── 24-AttentionVisualizer.py   # Attention heatmaps
│   ├── 25-ShapAnalyzer.py          # SHAP interpretability
│   ├── 26-PowerLawAnalyzer.py      # Power law analysis
│   ├── 27-ErrorAnalysis.py         # Comprehensive error analysis
│   ├── 28-Visualizer.py            # Basic plotting functions
│   ├── 29-ReportGenerator.py       # PDF report generation
│   ├── 30-RefusalPipeline.py       # Main training pipeline + WildJailbreak supplementation
│   ├── 31-ExperimentRunner.py      # Experiment orchestration
│   ├── 32-Execute.py               # Main entry point
│   ├── 33-Analyze.py               # Analysis script
│   ├── 34-ProductionAPI.py         # FastAPI server
│   ├── 35-MonitoringSystem.py      # Production monitoring
│   ├── 36-RetrainingPipeline.py    # Automated retraining
│   └── 37-DataManager.py           # Production data management
├── data/                           # Created automatically
├── models/                         # Created automatically
├── results/                        # Created automatically
├── visualizations/                 # Created automatically
├── reports/                        # Created automatically
├── requirements.txt                # Python dependencies
├── requirements-aws.txt            # AWS-specific dependencies
├── Dockerfile                      # Docker multi-stage build
├── docker-compose.yml              # Docker Compose configuration
├── .dockerignore                   # Docker ignore patterns
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

---

## 🚀 Quick Start

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/Dual-RoBERTa-Classifiers.git
cd Dual-RoBERTa-Classifiers

# Install dependencies
pip install -r requirements.txt
```

### **2. Set API Keys**

```bash
# Required API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# Optional (for AWS features)
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
```

### **3. Run Quick Test**

```bash
python src/32-Execute.py --test
```

### **4. Run Full Experiment**

```bash
# Interactive mode
python src/32-Execute.py

# CLI modes
python src/32-Execute.py --full         # Full pipeline
python src/32-Execute.py --train-only   # Training only
python src/32-Execute.py --analyze-only # Analysis only
python src/32-Execute.py --cv           # Cross-validation mode
python src/32-Execute.py --cv 10        # Cross-validation with 10 folds
```

---

## 🐳 Docker Deployment

### **Why Docker?**
Docker provides environment consistency, easy deployment, and reproducibility across different machines and environments.

### **Quick Start with Docker:**

```bash
# 1. Build the image
docker-compose build dev

# 2. Start development environment
docker-compose up dev

# Or use specific services:
docker-compose up train      # Full training pipeline
docker-compose up train-cv   # Cross-validation (Phase 2)
docker-compose up analyze    # Analysis only
docker-compose up api        # Production API server
docker-compose up jupyter    # Jupyter notebook
```

### **Docker Services Available:**

| Service | Purpose | Command |
|---------|---------|---------|
| `dev` | Interactive development | `docker-compose up dev` |
| `train` | Full training pipeline | `docker-compose up train` |
| `train-cv` | Cross-validation training | `docker-compose up train-cv` |
| `analyze` | Analysis with reports | `docker-compose up analyze` |
| `api` | Production API server | `docker-compose up api` |
| `jupyter` | Jupyter notebook | `docker-compose up jupyter` |

### **Environment Setup:**

Create a `.env` file in the project root with your API keys:

```bash
# .env file
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key

# Optional AWS credentials
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

### **GPU Support:**

Docker Compose is configured for GPU support. To use CPU-only:

```bash
# Comment out the deploy section in docker-compose.yml
# Or use CPU-only base image
docker build --target production -t refusal-classifier:cpu .
```

### **Common Docker Commands:**

```bash
# Build all images
docker-compose build

# Run full training pipeline (with GPU)
docker-compose up train

# Run cross-validation (Phase 2)
docker-compose up train-cv

# Start API server
docker-compose up -d api

# Access development container shell
docker-compose run --rm dev /bin/bash

# View API logs
docker-compose logs -f api

# Stop all services
docker-compose down

# Remove all containers and volumes
docker-compose down -v
```

### **Volume Mounts:**

Docker containers mount local directories for persistence:

- `./data` → `/app/data` - Training data
- `./models` → `/app/models` - Trained models
- `./results` → `/app/results` - Analysis results
- `./visualizations` → `/app/visualizations` - Plots
- `./reports` → `/app/reports` - PDF reports

**WHY:** Changes persist even when containers are stopped/restarted.

### **Production Deployment:**

```bash
# Build production API image
docker build --target api -t refusal-classifier:api .

# Run with Docker
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  --name refusal-api \
  refusal-classifier:api

# Or use Docker Compose
docker-compose up -d api

# Health check
curl http://localhost:8000/health
```

---

## 📊 Usage Examples

### **Training & Analysis:**

```bash
# Full experiment with all stages
python src/32-Execute.py --full

# Cross-validation with hypothesis testing and error analysis
python src/32-Execute.py --cv           # 5-fold CV (default)
python src/32-Execute.py --cv 10        # 10-fold CV

# Analyze existing models with PDF reports
python src/33-Analyze.py --auto --generate-report

# Specify custom models
python src/33-Analyze.py --refusal-model models/my_model.pt \
                         --jailbreak-model models/jailbreak.pt \
                         --generate-report --report-type performance
```

### **Production API:**

```bash
# Start production server
python src/34-ProductionAPI.py

# Test classification endpoint
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How do I hack a website?",
    "response": "I cannot help with that request."
  }'
```

---

## 🏗️ Architecture

### **Data Pipeline:**
```
Prompt Generation (3-stage) → Multi-LLM Response Collection → Data Cleaning → LLM Judge Labeling → Quality Analysis → Train/Val/Test Split
```

### **Training Pipeline:**
```
Load Data → Create Dataset → Initialize Model (RoBERTa) → Weighted Loss → Train with Early Stopping → Validation → Save Best Model
```

### **Analysis Pipeline:**
```
Load Model → Per-Model Analysis → Confidence Analysis → Adversarial Testing → Jailbreak Analysis → SHAP → Attention → Power Law → Visualizations → PDF Reports
```

### **Production Pipeline:**
```
FastAPI Server → Model Inference → Log to PostgreSQL → Monitor Performance → Detect Drift → Trigger Retraining → A/B Test → Deploy
```

---

## 📈 Model Performance

**Refusal Classifier (3-class):**
- Classes: No Refusal, Hard Refusal, Soft Refusal
- Model: RoBERTa-base fine-tuned
- Training: Weighted CrossEntropyLoss (class imbalance handling)
- Evaluation: F1-score (macro), Precision, Recall, Accuracy

**Jailbreak Detector (2-class):**
- Classes: Attack Failed, Attack Succeeded
- Model: RoBERTa-base fine-tuned
- Focus: Security-critical detection

*(Run experiments to get specific metrics)*

---

## 🔬 Interpretability

The project includes comprehensive interpretability tools:

1. **SHAP Analysis**: Token-level feature importance
2. **Attention Visualization**: Multi-head attention heatmaps
3. **Power Law Analysis**: Pareto principle in predictions
4. **Confidence Analysis**: Calibration and uncertainty
5. **Adversarial Robustness**: Paraphrasing sensitivity

---

## 📦 Configuration

All settings are centralized in `src/04-Config.py` (the Control Room):

- **API Configuration**: Models, rate limits, retries
- **Dataset Configuration**: Sample sizes, categories
- **Training Configuration**: Epochs, batch size, learning rate
- **Model Configuration**: Architecture, num_classes, dropout
- **Production Configuration**: Monitoring, A/B testing, retraining
- **WildJailbreak Configuration**: Dataset supplementation settings

---

## 🧪 Testing

```bash
# Quick test (reduced samples)
python src/32-Execute.py --test

# Train only (uses existing data)
python src/32-Execute.py --train-only

# Analyze only (uses existing models)
python src/32-Execute.py --analyze-only
```

---

## 📄 Reports

Generate professional PDF reports:

```bash
# All reports (performance + interpretability + executive summary)
python src/33-Analyze.py --auto --generate-report --report-type all

# Performance report only
python src/33-Analyze.py --auto --generate-report --report-type performance

# Executive summary only
python src/33-Analyze.py --auto --generate-report --report-type executive
```

Reports are saved to `reports/` directory.

---

## 🔐 Security

**Important:** Never commit API keys or sensitive data!

- All API keys via environment variables
- AWS Secrets Manager integration available
- `.gitignore` configured to exclude secrets
- Production API requires admin key configuration

---

## 🛠️ Development

### **File Numbering Convention:**
Files are numbered 01-37 for clear execution order.

### **Generic Design:**
All analyzers and visualizers use `class_names` parameter - works with any N-class classifier.

### **Adding New Features:**
1. Add module to `src/`
2. Follow naming convention: `XX-ModuleName.py`
3. Include header comment describing the module
4. Update `01-Imports.py` if new dependencies are needed
5. Update `04-Config.py` for any new configuration parameters

---

## 📚 Dependencies

See `requirements.txt` for full list. Key dependencies:

- **PyTorch** (2.0+): Deep learning framework
- **Transformers** (4.30+): Hugging Face models
- **FastAPI** (0.100+): Production API
- **ReportLab** (4.0+): PDF generation
- **SHAP** (0.42+): Interpretability

---

## 🤝 Contributing

This is a research/production project. For questions or suggestions, please open an issue.

---

## 📝 License

[Specify your license - see LICENSE file]

---

## 👤 Author

Ramy Alsaffar

---

## 🙏 Acknowledgments

### **Models & Libraries:**
- **RoBERTa**: Liu et al., 2019 - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- **SHAP**: Lundberg & Lee, 2017 - [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- **Transformers**: Hugging Face team - [State-of-the-art NLP](https://github.com/huggingface/transformers)
- **FastAPI**: Sebastián Ramírez - [Modern web framework](https://fastapi.tiangolo.com/)

### **Datasets:**
- **WildJailbreak**: AllenAI, 2024 - Used for supplementing jailbreak detection training data
  - Paper: [WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models](https://arxiv.org/abs/2406.18510)
  - Authors: Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, Nouha Dziri
  - Dataset: [allenai/wildjailbreak on Hugging Face](https://huggingface.co/datasets/allenai/wildjailbreak)
  - License: Apache 2.0
  - 262K prompt-response pairs with 82,728 adversarial harmful samples
  - This project uses WildJailbreak's adversarial harmful subset for supplementing jailbreak detection training when insufficient positive samples are collected from our primary pipeline

---

## 📧 Contact

[Your contact information]

---

**Last Updated:** November 3, 2025
