# ReportGenerator Module
#-----------------------
# Generates comprehensive PDF reports using ReportLab.
# Creates professional reports for model performance, interpretability,
# production monitoring, and executive summaries.
# Requirements: pip install reportlab
# All imports are in 01-Imports.py
###############################################################################


class ReportGenerator:
    """
    GENERIC: Generates professional PDF reports for any N-class classifier.

    Supports multiple report types:
    - Model Performance Report
    - Jailbreak Security Report
    - Production Monitoring Report
    - Interpretability Report
    - Executive Summary
    """

    def __init__(self, class_names: List[str] = None):
        """
        Initialize the report generator.

        Args:
            class_names: List of class names (default: CLASS_NAMES from config)
        """
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create custom paragraph styles for reports."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))

        # Subsection heading
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#d62728'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        ))

        # Metric style (for key numbers)
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#ff7f0e'),
            fontName='Helvetica-Bold'
        ))

        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        ))

    def _fig_to_image(self, fig, width: float = 6*inch) -> Image:
        """
        Convert matplotlib figure to reportlab Image.

        Args:
            fig: Matplotlib figure
            width: Width in reportlab units

        Returns:
            ReportLab Image object
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image(buf, width=width)
        plt.close(fig)
        return img

    def _create_header(self, title: str, subtitle: str = None) -> List:
        """Create report header with title and subtitle."""
        elements = []
        elements.append(Paragraph(title, self.styles['CustomTitle']))

        if subtitle:
            elements.append(Paragraph(subtitle, self.styles['Normal']))
            elements.append(Spacer(1, 12))

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(
            f"<i>Generated: {timestamp}</i>",
            self.styles['Footer']
        ))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1f77b4')))
        elements.append(Spacer(1, 20))

        return elements

    def _create_data_composition_section(self, data_composition_stats: Dict) -> List:
        """
        Create data composition section for WildJailbreak supplementation reporting.

        Args:
            data_composition_stats: Dictionary with keys:
                - real_count: Number of real samples
                - wildjailbreak_count: Number of WildJailbreak samples
                - total_count: Total samples
                - real_percentage: Percentage of real samples
                - wildjailbreak_percentage: Percentage of WildJailbreak samples
                - supplementation_used: Boolean indicating if supplementation was used

        Returns:
            List of ReportLab elements
        """
        elements = []

        if not data_composition_stats or not data_composition_stats.get('supplementation_used', False):
            # No supplementation used - all real data
            elements.append(Paragraph("Training Data Composition", self.styles['SectionHeading']))
            elements.append(Paragraph(
                "All training data was collected from real model responses. "
                "No supplementation from external datasets was required.",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 12))
            return elements

        # Supplementation was used
        elements.append(Paragraph("Training Data Composition", self.styles['SectionHeading']))

        real_count = data_composition_stats.get('real_count', 0)
        wildjailbreak_count = data_composition_stats.get('wildjailbreak_count', 0)
        total_count = data_composition_stats.get('total_count', real_count + wildjailbreak_count)
        real_pct = data_composition_stats.get('real_percentage', (real_count / total_count * 100) if total_count > 0 else 0)
        wild_pct = data_composition_stats.get('wildjailbreak_percentage', (wildjailbreak_count / total_count * 100) if total_count > 0 else 0)

        composition_text = f"""
        Training data was supplemented with samples from the <b>WildJailbreak dataset</b> to ensure
        sufficient positive samples for jailbreak detection. Modern LLMs (Claude Sonnet 4.5, GPT-5,
        Gemini 2.5 Flash) successfully defended against all jailbreak attempts in initial testing,
        requiring external supplementation.
        <br/><br/>
        <b>Data Composition:</b>
        <br/>• Real model responses: <b>{real_count:,} samples ({real_pct:.1f}%)</b>
        <br/>• WildJailbreak samples: <b>{wildjailbreak_count:,} samples ({wild_pct:.1f}%)</b>
        <br/>• Total training samples: <b>{total_count:,}</b>
        """

        elements.append(Paragraph(composition_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))

        # Add citation if WildJailbreak was used
        if wildjailbreak_count > 0:
            elements.extend(self._create_wildjailbreak_citation())

        return elements

    def _create_wildjailbreak_citation(self) -> List:
        """
        Create WildJailbreak dataset citation section.

        Returns:
            List of ReportLab elements
        """
        elements = []

        elements.append(Paragraph("WildJailbreak Dataset Attribution", self.styles['SubsectionHeading']))

        citation_text = f"""
        <b>Dataset:</b> {WILDJAILBREAK_DATASET_INFO['name']} by {WILDJAILBREAK_DATASET_INFO['source']}
        <br/><b>Paper:</b> {WILDJAILBREAK_DATASET_INFO['paper_title']}
        <br/><b>Authors:</b> {WILDJAILBREAK_DATASET_INFO['authors']}
        <br/><b>Conference:</b> {WILDJAILBREAK_DATASET_INFO['conference']} {WILDJAILBREAK_DATASET_INFO['year']}
        <br/><b>Dataset Size:</b> {WILDJAILBREAK_DATASET_INFO['size']}
        <br/><b>License:</b> {WILDJAILBREAK_DATASET_INFO['license']}
        <br/><b>URL:</b> <link href="{WILDJAILBREAK_DATASET_INFO['url']}">{WILDJAILBREAK_DATASET_INFO['url']}</link>
        <br/><br/>
        <i>This project uses WildJailbreak's adversarial harmful subset ({WILDJAILBREAK_DATASET_INFO['adversarial_harmful_samples']} samples)
        for supplementing jailbreak detection training when insufficient positive samples are collected from our primary pipeline.</i>
        """

        elements.append(Paragraph(citation_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))

        return elements

    def _create_metrics_table(self, metrics_dict: Dict, title: str = None) -> List:
        """
        Create a formatted table from metrics dictionary.

        Args:
            metrics_dict: Dictionary of metric name -> value
            title: Optional table title

        Returns:
            List of flowable elements
        """
        elements = []

        if title:
            elements.append(Paragraph(title, self.styles['SubsectionHeading']))

        # Build table data
        data = [['Metric', 'Value']]
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            data.append([key, value_str])

        # Create table
        table = Table(data, colWidths=[3.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))

        elements.append(table)
        elements.append(Spacer(1, 12))

        return elements

    def generate_model_performance_report(
        self,
        model_name: str,
        metrics: Dict,
        confusion_matrix_fig,
        training_curves_fig,
        class_distribution_fig,
        output_path: str,
        data_composition_stats: Dict = None
    ):
        """
        Generate comprehensive model performance report.

        Args:
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            confusion_matrix_fig: Matplotlib figure of confusion matrix
            training_curves_fig: Matplotlib figure of training curves
            class_distribution_fig: Matplotlib figure of class distribution
            output_path: Path to save PDF report
            data_composition_stats: Optional dict with WildJailbreak supplementation stats (NEW - V09)
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []

        # Header
        elements.extend(self._create_header(
            f"{model_name} Performance Report",
            f"Comprehensive analysis of {self.num_classes}-class classification model"
        ))

        # Executive Summary
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeading']))
        summary_text = f"""
        This report presents a detailed performance analysis of the <b>{model_name}</b> classifier.
        The model classifies text into {self.num_classes} categories: {', '.join(self.class_names)}.
        Overall accuracy: <b>{metrics.get('accuracy', 0):.2%}</b>,
        Macro F1 Score: <b>{metrics.get('macro_f1', 0):.4f}</b>.
        """
        elements.append(Paragraph(summary_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))

        # Data Composition Section (NEW - V09)
        if data_composition_stats:
            elements.extend(self._create_data_composition_section(data_composition_stats))
            elements.append(Spacer(1, 12))

        # Overall Metrics
        overall_metrics = {
            'Accuracy': metrics.get('accuracy', 0),
            'Macro F1': metrics.get('macro_f1', 0),
            'Weighted F1': metrics.get('weighted_f1', 0),
            'Macro Precision': metrics.get('macro_precision', 0),
            'Macro Recall': metrics.get('macro_recall', 0),
        }
        elements.extend(self._create_metrics_table(overall_metrics, "Overall Performance Metrics"))

        # Per-class metrics
        elements.append(PageBreak())
        elements.append(Paragraph("Per-Class Performance", self.styles['SectionHeading']))

        for i, class_name in enumerate(self.class_names):
            class_metrics = {
                'Precision': metrics.get(f'class_{i}_precision', 0),
                'Recall': metrics.get(f'class_{i}_recall', 0),
                'F1 Score': metrics.get(f'class_{i}_f1', 0),
                'Support': metrics.get(f'class_{i}_support', 0),
            }
            elements.extend(self._create_metrics_table(class_metrics, f"Class: {class_name}"))

        # Confusion Matrix
        elements.append(PageBreak())
        elements.append(Paragraph("Confusion Matrix", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "The confusion matrix shows the model's predictions versus actual labels. "
            "Diagonal elements represent correct predictions.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(confusion_matrix_fig))

        # Training Curves
        elements.append(PageBreak())
        elements.append(Paragraph("Training Curves", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Training and validation metrics over epochs. "
            "Monitor for overfitting (validation diverging from training).",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(training_curves_fig))

        # Class Distribution
        elements.append(PageBreak())
        elements.append(Paragraph("Class Distribution", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Distribution of samples across classes. "
            "Imbalanced datasets may require weighted loss or resampling.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(class_distribution_fig))

        # Build PDF
        doc.build(elements)
        print(f"✓ Model Performance Report saved to: {output_path}")

    def generate_interpretability_report(
        self,
        model_name: str,
        attention_figs: List,
        shap_figs: List,
        power_law_figs: List,
        sample_texts: List[str],
        output_path: str
    ):
        """
        Generate interpretability report with SHAP, attention, and power law analysis.

        Args:
            model_name: Name of the model
            attention_figs: List of attention visualization figures
            shap_figs: List of SHAP analysis figures
            power_law_figs: List of power law analysis figures
            sample_texts: List of sample texts analyzed
            output_path: Path to save PDF report
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []

        # Header
        elements.extend(self._create_header(
            f"{model_name} Interpretability Report",
            "Understanding model decisions through attention, SHAP, and power law analysis"
        ))

        # Introduction
        elements.append(Paragraph("Introduction", self.styles['SectionHeading']))
        intro_text = """
        This report provides insights into how the model makes predictions.
        We analyze attention patterns, feature importance via SHAP values,
        and power law distributions in model behavior.
        """
        elements.append(Paragraph(intro_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))

        # Attention Analysis
        if attention_figs:
            elements.append(PageBreak())
            elements.append(Paragraph("Attention Analysis", self.styles['SectionHeading']))
            elements.append(Paragraph(
                "Attention heatmaps show which tokens the model focuses on when making predictions. "
                "Brighter colors indicate higher attention weights.",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 12))

            for i, (fig, text) in enumerate(zip(attention_figs, sample_texts[:len(attention_figs)])):
                elements.append(Paragraph(f"Sample {i+1}: {text[:100]}...", self.styles['SubsectionHeading']))
                elements.append(self._fig_to_image(fig))
                elements.append(Spacer(1, 12))

        # SHAP Analysis
        if shap_figs:
            elements.append(PageBreak())
            elements.append(Paragraph("SHAP Analysis", self.styles['SectionHeading']))
            elements.append(Paragraph(
                "SHAP (SHapley Additive exPlanations) values show each token's contribution "
                "to the model's prediction. Red indicates positive contribution, blue indicates negative.",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 12))

            for i, fig in enumerate(shap_figs):
                elements.append(Paragraph(f"SHAP Analysis {i+1}", self.styles['SubsectionHeading']))
                elements.append(self._fig_to_image(fig))
                elements.append(Spacer(1, 12))

        # Power Law Analysis
        if power_law_figs:
            elements.append(PageBreak())
            elements.append(Paragraph("Power Law Analysis", self.styles['SectionHeading']))
            elements.append(Paragraph(
                "Power law analysis reveals Pareto distributions in model behavior, "
                "showing if a small subset of features drives most predictions.",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 12))

            for i, fig in enumerate(power_law_figs):
                elements.append(self._fig_to_image(fig))
                elements.append(Spacer(1, 12))

        # Build PDF
        doc.build(elements)
        print(f"✓ Interpretability Report saved to: {output_path}")

    def generate_production_monitoring_report(
        self,
        predictions_df: pd.DataFrame,
        metrics_over_time_fig,
        confidence_distribution_fig,
        latency_distribution_fig,
        ab_test_comparison_fig = None,
        output_path: str = None
    ):
        """
        Generate production monitoring report from logged predictions.

        Args:
            predictions_df: DataFrame with prediction logs
            metrics_over_time_fig: Figure showing metrics over time
            confidence_distribution_fig: Figure showing confidence distributions
            latency_distribution_fig: Figure showing prediction latency
            ab_test_comparison_fig: Optional A/B test comparison figure
            output_path: Path to save PDF report
        """
        if output_path is None:
            output_path = os.path.join(reports_path, f"production_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []

        # Header
        elements.extend(self._create_header(
            "Production Monitoring Report",
            f"Analysis of {len(predictions_df)} predictions"
        ))

        # Summary Statistics
        elements.append(Paragraph("Summary Statistics", self.styles['SectionHeading']))

        total_predictions = len(predictions_df)
        avg_confidence = predictions_df['confidence'].mean()
        avg_latency = predictions_df['latency_ms'].mean()

        # Predictions per class
        class_counts = {}
        for i, class_name in enumerate(self.class_names):
            count = (predictions_df['prediction'] == i).sum()
            class_counts[f"{class_name} predictions"] = count

        summary_metrics = {
            'Total Predictions': total_predictions,
            'Average Confidence': f"{avg_confidence:.4f}",
            'Average Latency (ms)': f"{avg_latency:.2f}",
            **class_counts
        }

        elements.extend(self._create_metrics_table(summary_metrics))

        # Metrics Over Time
        elements.append(PageBreak())
        elements.append(Paragraph("Metrics Over Time", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Monitor for model drift and performance degradation over time.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(metrics_over_time_fig))

        # Confidence Distribution
        elements.append(PageBreak())
        elements.append(Paragraph("Confidence Distribution", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Distribution of prediction confidence scores. "
            "Low confidence predictions may require human review.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(confidence_distribution_fig))

        # Latency Analysis
        elements.append(PageBreak())
        elements.append(Paragraph("Latency Analysis", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Prediction latency distribution. Monitor for performance bottlenecks.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(latency_distribution_fig))

        # A/B Test Comparison (if available)
        if ab_test_comparison_fig is not None:
            elements.append(PageBreak())
            elements.append(Paragraph("A/B Test Comparison", self.styles['SectionHeading']))
            elements.append(Paragraph(
                "Performance comparison between active and challenger models.",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 12))
            elements.append(self._fig_to_image(ab_test_comparison_fig))

        # Build PDF
        doc.build(elements)
        print(f"✓ Production Monitoring Report saved to: {output_path}")
        return output_path

    def generate_executive_summary(
        self,
        model_name: str,
        key_metrics: Dict,
        performance_chart_fig,
        recommendations: List[str],
        output_path: str,
        data_composition_stats: Dict = None
    ):
        """
        Generate 1-2 page executive summary for stakeholders.

        Args:
            model_name: Name of the model
            key_metrics: Dictionary of key performance indicators
            performance_chart_fig: Summary performance visualization
            recommendations: List of actionable recommendations
            output_path: Path to save PDF report
            data_composition_stats: Optional dict with WildJailbreak supplementation stats (NEW - V09)
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []

        # Header
        elements.extend(self._create_header(
            f"{model_name} - Executive Summary",
            "High-level performance overview and recommendations"
        ))

        # Data Composition (if applicable) (NEW - V09)
        if data_composition_stats and data_composition_stats.get('supplementation_used', False):
            elements.extend(self._create_data_composition_section(data_composition_stats))
            elements.append(Spacer(1, 12))

        # Key Metrics
        elements.append(Paragraph("Key Performance Indicators", self.styles['SectionHeading']))
        elements.extend(self._create_metrics_table(key_metrics))

        # Performance Chart
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(performance_chart_fig))

        # Recommendations
        elements.append(PageBreak())
        elements.append(Paragraph("Recommendations", self.styles['SectionHeading']))

        for i, rec in enumerate(recommendations, 1):
            elements.append(Paragraph(f"{i}. {rec}", self.styles['BodyText']))
            elements.append(Spacer(1, 8))

        # Build PDF
        doc.build(elements)
        print(f"✓ Executive Summary saved to: {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 31, 2025
@author: ramyalsaffar
"""
