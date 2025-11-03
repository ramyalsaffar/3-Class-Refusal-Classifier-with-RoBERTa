# Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection

A production-ready, dual-task classification system using fine-tuned RoBERTa models for comprehensive LLM safety analysis:
1. **Refusal Classification** (3 classes): No Refusal, Hard Refusal, Soft Refusal
2. **Jailbreak Detection** (2 classes): Attack Failed, Attack Succeeded

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
│   ├── 01-Imports.py              # Central import manager
│   ├── 02-Config.py                # All configuration settings (includes AWS config)
│   ├── 03-Constants.py             # Global constants
│   ├── 04-SecretsHandler.py        # AWS Secrets Manager
│   ├── 05-PromptGenerator.py       # 3-stage prompt generation
│   ├── 06-ResponseCollector.py     # Multi-LLM response collection
│   ├── 07-DataCleaner.py           # Comprehensive data cleaning
│   ├── 08-DataLabeler.py           # LLM judge labeling
│   ├── 09-LabelingQualityAnalyzer.py
│   ├── 10-ClassificationDataset.py # PyTorch Dataset
│   ├── 11-RefusalClassifier.py     # 3-class RoBERTa model
│   ├── 12-JailbreakDetector.py     # 2-class RoBERTa model
│   ├── 13-Trainer.py               # Standard trainer with weighted loss
│   ├── 14-CrossValidator.py        # K-fold cross-validation (Phase 2)
│   ├── 15-PerModelAnalyzer.py      # Per-model performance analysis
│   ├── 16-ConfidenceAnalyzer.py    # Confidence score analysis
│   ├── 17-AdversarialTester.py     # Paraphrasing robustness tests
│   ├── 18-JailbreakAnalysis.py     # Security-focused jailbreak analysis
│   ├── 19-CorrelationAnalysis.py   # Refusal ↔ Jailbreak correlation (Phase 2)
│   ├── 20-AttentionVisualizer.py   # Attention heatmaps
│   ├── 21-ShapAnalyzer.py          # SHAP interpretability
│   ├── 22-PowerLawAnalyzer.py      # Power law analysis
│   ├── 23-HypothesisTesting.py     # Statistical hypothesis tests (Phase 2)
│   ├── 24-ErrorAnalysis.py         # Comprehensive error analysis (Phase 2)
│   ├── 25-Visualizer.py            # Basic plotting functions
│   ├── 26-ReportGenerator.py       # PDF report generation
│   ├── 27-RefusalPipeline.py       # Main training pipeline
│   ├── 28-ExperimentRunner.py      # Experiment orchestration
│   ├── 29-Execute.py               # Main entry point
│   ├── 30-Analyze.py               # Analysis script
│   ├── 31-ProductionAPI.py         # FastAPI server
│   ├── 32-MonitoringSystem.py      # Production monitoring
│   ├── 33-RetrainingPipeline.py    # Automated retraining
│   └── 34-DataManager.py           # Production data management
├── data/                           # Created automatically
├── models/                         # Created automatically
├── results/                        # Created automatically
├── visualizations/                 # Created automatically
├── reports/                        # Created automatically
├── requirements.txt                # Python dependencies
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
python src/29-Execute.py --test
```

### **4. Run Full Experiment**

```bash
# Interactive mode
python src/29-Execute.py

# CLI modes
python src/29-Execute.py --full         # Full pipeline
python src/29-Execute.py --train-only   # Training only
python src/29-Execute.py --analyze-only # Analysis only
python src/29-Execute.py --cv           # Cross-validation mode (Phase 2)
python src/29-Execute.py --cv 10        # Cross-validation with 10 folds
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
python src/29-Execute.py --full

# Cross-validation with hypothesis testing and error analysis (Phase 2)
python src/29-Execute.py --cv           # 5-fold CV (default)
python src/29-Execute.py --cv 10        # 10-fold CV

# Analyze existing models with PDF reports
python src/30-Analyze.py --auto --generate-report

# Specify custom models
python src/30-Analyze.py --refusal-model models/my_model.pt \
                         --jailbreak-model models/jailbreak.pt \
                         --generate-report --report-type performance
```

### **Production API:**

```bash
# Start production server
python src/31-ProductionAPI.py

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

All settings are centralized in `src/02-Config.py`:

- **API Configuration**: Models, rate limits, retries
- **Dataset Configuration**: Sample sizes, categories
- **Training Configuration**: Epochs, batch size, learning rate
- **Model Configuration**: Architecture, num_classes, dropout
- **Production Configuration**: Monitoring, A/B testing, retraining

---

## 🧪 Testing

```bash
# Quick test (reduced samples)
python src/29-Execute.py --test

# Train only (uses existing data)
python src/29-Execute.py --train-only

# Analyze only (uses existing models)
python src/29-Execute.py --analyze-only
```

---

## 📄 Reports

Generate professional PDF reports:

```bash
# All reports (performance + interpretability + executive summary)
python src/30-Analyze.py --auto --generate-report --report-type all

# Performance report only
python src/30-Analyze.py --auto --generate-report --report-type performance

# Executive summary only
python src/30-Analyze.py --auto --generate-report --report-type executive
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
Files are numbered 01-31 for clear execution order. Auto-loaded files: 04-25.

### **Generic Design:**
All analyzers and visualizers use `class_names` parameter - works with any N-class classifier.

### **Adding New Features:**
1. Add module to `src/`
2. Follow naming convention: `XX-ModuleName.py`
3. Include header: `# All imports are in 01-Imports.py`
4. Update `01-Imports.py` if needed

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

- **RoBERTa**: Liu et al., 2019
- **SHAP**: Lundberg & Lee, 2017
- **Transformers**: Hugging Face team
- **FastAPI**: Sebastián Ramírez

---

## 📧 Contact

[Your contact information]

---

**Last Updated:** November 3, 2025
