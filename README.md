# 3-Class Refusal Classifier with RoBERTa

A production-ready, fine-tuned RoBERTa model for detecting refusal patterns in Large Language Model (LLM) responses with **dual-task classification**:
1. **Refusal Classification** (3 classes): No Refusal, Hard Refusal, Soft Refusal
2. **Jailbreak Detection** (2 classes): Attack Failed, Attack Succeeded

---

## 🎯 Features

### **Core Capabilities:**
- ✅ **Dual-Task Classification**: Simultaneously detects refusal patterns AND jailbreak success
- ✅ **Production-Ready**: Full API server with A/B testing, monitoring, and auto-retraining
- ✅ **Comprehensive Interpretability**: SHAP analysis, attention visualization, power law analysis
- ✅ **LLM Judge Labeling**: GPT-4-based labeling with confidence scores and randomized class ordering
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
3-Class-Refusal-Classifier-with-RoBERTa/
├── src/
│   ├── 01-Imports.py              # Central import manager
│   ├── 02-Config.py                # All configuration settings
│   ├── 03-Constants.py             # Global constants
│   ├── 04-AWSConfig.py             # AWS configuration
│   ├── 05-SecretsHandler.py        # AWS Secrets Manager
│   ├── 06-PromptGenerator.py       # 3-stage prompt generation
│   ├── 07-ResponseCollector.py     # Multi-LLM response collection
│   ├── 08-DataCleaner.py           # Comprehensive data cleaning
│   ├── 09-DataLabeler.py           # LLM judge labeling
│   ├── 10-LabelingQualityAnalyzer.py
│   ├── 11-ClassificationDataset.py # PyTorch Dataset
│   ├── 12-RefusalClassifier.py     # 3-class RoBERTa model
│   ├── 13-JailbreakDetector.py     # 2-class RoBERTa model
│   ├── 14-Trainer.py               # Generic trainer with weighted loss
│   ├── 15-PerModelAnalyzer.py      # Per-model performance analysis
│   ├── 16-ConfidenceAnalyzer.py    # Confidence score analysis
│   ├── 17-AdversarialTester.py     # Paraphrasing robustness tests
│   ├── 18-JailbreakAnalysis.py     # Security analysis
│   ├── 19-AttentionVisualizer.py   # Attention heatmaps
│   ├── 20-ShapAnalyzer.py          # SHAP interpretability
│   ├── 21-PowerLawAnalyzer.py      # Power law analysis
│   ├── 22-Visualizer.py            # Basic plotting functions
│   ├── 23-ReportGenerator.py       # PDF report generation
│   ├── 24-RefusalPipeline.py       # Main training pipeline
│   ├── 25-ExperimentRunner.py      # Experiment orchestration
│   ├── 26-Execute.py               # Main entry point
│   ├── 27-Analyze.py               # Analysis script
│   ├── 28-ProductionAPI.py         # FastAPI server
│   ├── 29-MonitoringSystem.py      # Production monitoring
│   ├── 30-RetrainingPipeline.py    # Automated retraining
│   ├── 31-DataManager.py           # Production data management
│   ├── 32-CrossValidator.py        # K-fold cross-validation (Phase 2)
│   ├── 33-HypothesisTesting.py     # Statistical hypothesis tests (Phase 2)
│   └── 34-ErrorAnalysis.py         # Comprehensive error analysis (Phase 2)
├── data/                           # Created automatically
├── models/                         # Created automatically
├── results/                        # Created automatically
├── visualizations/                 # Created automatically
├── reports/                        # Created automatically
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

---

## 🚀 Quick Start

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/3-Class-Refusal-Classifier-with-RoBERTa.git
cd 3-Class-Refusal-Classifier-with-RoBERTa

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
python src/26-Execute.py --test
```

### **4. Run Full Experiment**

```bash
# Interactive mode
python src/26-Execute.py

# CLI modes
python src/26-Execute.py --full         # Full pipeline
python src/26-Execute.py --train-only   # Training only
python src/26-Execute.py --analyze-only # Analysis only
python src/26-Execute.py --cv           # Cross-validation mode (Phase 2)
python src/26-Execute.py --cv 10        # Cross-validation with 10 folds
```

---

## 📊 Usage Examples

### **Training & Analysis:**

```bash
# Full experiment with all stages
python src/26-Execute.py --full

# Cross-validation with hypothesis testing and error analysis (Phase 2)
python src/26-Execute.py --cv           # 5-fold CV (default)
python src/26-Execute.py --cv 10        # 10-fold CV

# Analyze existing models with PDF reports
python src/27-Analyze.py --auto --generate-report

# Specify custom models
python src/27-Analyze.py --refusal-model models/my_model.pt \
                         --jailbreak-model models/jailbreak.pt \
                         --generate-report --report-type performance
```

### **Production API:**

```bash
# Start production server
python src/28-ProductionAPI.py

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
python src/26-Execute.py --test

# Train only (uses existing data)
python src/26-Execute.py --train-only

# Analyze only (uses existing models)
python src/26-Execute.py --analyze-only
```

---

## 📄 Reports

Generate professional PDF reports:

```bash
# All reports (performance + interpretability + executive summary)
python src/27-Analyze.py --auto --generate-report --report-type all

# Performance report only
python src/27-Analyze.py --auto --generate-report --report-type performance

# Executive summary only
python src/27-Analyze.py --auto --generate-report --report-type executive
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

**Last Updated:** October 31, 2025
