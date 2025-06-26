import pandas as pd
import numpy as np
import joblib
import streamlit as st
import requests
import logging
from lime.lime_tabular import LimeTabularExplainer
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MECP2Classifier:
    def __init__(self, model_path: str, gtf_path: Optional[str] = None):
        self.model = self.load_model(model_path)
        self.exon_coords = []
        if gtf_path:
            self.load_gtf(gtf_path)

        self.mc_labels = {
            "synonymous variant": "mc_synonymous_variant",
            "3 prime UTR variant": "mc_3_prime_UTR_variant",
            "5 prime UTR variant": "mc_5_prime_UTR_variant",
            "splice donor variant": "mc_splice_donor_variant",
            "splice acceptor variant": "mc_splice_acceptor_variant",
            "nonsense": "mc_nonsense",
            "intron variant": "mc_intron_variant",
            "missense variant": "mc_missense_variant",
            "stop lost": "mc_stop_lost",
            "frameshift variant": "mc_frameshift_variant"
        }

        self.expected_features = [
        'position', 'alignment_score', 'mc_synonymous_variant', 'mc_frameshift_variant',
        'mc_3_prime_UTR_variant', 'mc_5_prime_UTR_variant', 'mc_splice_donor_variant',
        'mc_splice_acceptor_variant', 'mc_nonsense', 'mc_intron_variant',
        'mc_missense_variant', 'mc_stop_lost', 'donor_distance', 'acceptor_distance',
        'dist_to_exon_start', 'dist_to_exon_end', 'region_exon',
        'region_non-exon', 'splice_type_acceptor', 'splice_type_donor', 'type',
        'prev_A', 'prev_C', 'prev_G', 'prev_T', 'next_A', 'next_C', 'next_G', 'next_T',
        'gc_content', 'gc_skew', 'at_content', 'position_bin', 'position_decile',
        'chr_chrX', 'hotspot_flag', 'cpg_overlap'
        ]

        self.gene_config = {
            'splice_sites': [154021750, 154021870, 154021980],
            'region_start': 154021500,
            'region_end': 154022200,
            'position_bins': 10
        }

        # Initialize LIME explainer (will be set up after first prediction)
        self.lime_explainer = None
        self.training_data = None

    def load_model(self, path: str):
        model = joblib.load(path)
        if not hasattr(model, 'predict'):
            raise ValueError("❌ Invalid model: missing predict()")
        return model

    def load_gtf(self, gtf_path: str):
        df = pd.read_csv(gtf_path, sep="\t", comment="#", header=None,
                         names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"])
        exon_df = df[df['feature'] == 'exon']
        self.exon_coords = list(zip(exon_df['seqname'], exon_df['start'], exon_df['end']))
        logger.info(f"Loaded {len(self.exon_coords)} exons from GTF")

    def is_exon(self, pos: int, chrom='X') -> bool:
        return any(c in ['X', 'chrX', '23'] and s <= pos <= e for c, s, e in self.exon_coords)

    def fetch_sequence_window(self, pos: int, window=20) -> str:
        start = max(1, pos - window // 2)
        end = pos + window // 2
        url = f"https://rest.ensembl.org/sequence/region/human/X:{start}..{end}?content-type=text/plain"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.text.strip().upper()
        except:
            logger.warning(f"Failed to fetch sequence for position {pos}")
        return "N" * window

    def compute_features(self, pos: int, deleted: str, inserted: str, consequences: List[str]) -> Tuple[pd.DataFrame, Tuple]:
        features = {key: 0 for key in self.expected_features}

        # 1. Positional Features
        features["position"] = pos

        # 2. Splice distance
        features["splice_distance"] = min(abs(pos - s) for s in self.gene_config['splice_sites'])

        # 3. Donor and Acceptor Distance
        donor_sites = [s for s in self.gene_config['splice_sites']]  # Use your real donor list if available
        acceptor_sites = [s for s in self.gene_config['splice_sites']]  # Use your real acceptor list if available
        features["donor_distance"] = min(abs(pos - s) for s in donor_sites)
        features["acceptor_distance"] = min(abs(pos - s) for s in acceptor_sites)

        # 4. Exon Region Features
        is_exon = self.is_exon(pos)
        features["region_exon"] = int(is_exon)
        features["region_non-exon"] = 1 - features["region_exon"]

        # 5. Distance to Exon Start and End
        exon_dists = []
        for c, start, end in self.exon_coords:
            if start <= pos <= end:
                exon_dists.append((abs(pos - start), abs(pos - end)))
        if exon_dists:
            dist_start, dist_end = exon_dists[0]
            features["dist_to_exon_start"] = dist_start
            features["dist_to_exon_end"] = dist_end
        else:
            features["dist_to_exon_start"] = -1
            features["dist_to_exon_end"] = -1

        # 6. Position Bin and Decile
        features["position_bin"] = int(np.digitize(pos, np.linspace(
            self.gene_config['region_start'], self.gene_config['region_end'], 11)))
        features["position_decile"] = features["position_bin"]

        # 7. Chromosome Binary
        features["chr_chrX"] = 1  # Because MECP2 is on X chromosome

        # 8. Variant Type Encoding
        if len(deleted) == 1 and len(inserted) == 1:
            features["type"] = 0  # SNV
        elif len(deleted) > 1 and inserted == "-":
            features["type"] = 1  # Deletion
        elif len(inserted) > 1 and deleted == "-":
            features["type"] = 2  # Insertion
        else:
            features["type"] = 3  # Complex/Indel

        # 9. Sequence Window
        window_seq = self.fetch_sequence_window(pos)

        # 10. GC Content
        def gc_content(s): return round((s.count("G") + s.count("C")) / len(s), 3)
        gc_ref = gc_content(window_seq)
        features["gc_content"] = gc_ref

        # 11. GC Skew
        g_count = window_seq.count('G')
        c_count = window_seq.count('C')
        features["gc_skew"] = round((g_count - c_count) / (g_count + c_count + 1e-5), 3)

        # 12. AT Content
        features["at_content"] = round((window_seq.count('A') + window_seq.count('T')) / len(window_seq), 3)

        # 13. Flanking Bases (prev and next nucleotide binary)
        mid = len(window_seq) // 2
        if mid > 0:
            prev_base = window_seq[mid - 1]
            if prev_base in ['A', 'C', 'G', 'T']:
                features[f"prev_{prev_base}"] = 1
        if mid < len(window_seq) - 1:
            next_base = window_seq[mid + 1]
            if next_base in ['A', 'C', 'G', 'T']:
                features[f"next_{next_base}"] = 1

        # 14. Alignment Score (compare ref vs mutated window)
        mut_window = list(window_seq)
        if inserted and inserted != "-" and mid < len(mut_window):
            mut_window[mid] = inserted
        mut_window = ''.join(mut_window)
        align_score = sum(1 for a, b in zip(window_seq, mut_window) if a == b)
        features["alignment_score"] = align_score

        # 15. Molecular Consequences (one-hot)
        for c in consequences:
            if c in self.mc_labels:
                features[self.mc_labels[c]] = 1

        # 16. Splice Type Binary Encoding
        features["splice_type_acceptor"] = int(features["splice_distance"] < 10)  # You can adjust threshold
        features["splice_type_donor"] = int(features["splice_distance"] < 10)

        # 17. Hotspot Flag (dummy example - replace with your real hotspot positions)
        hotspot_positions = [154021800, 154021900]  # Replace with actual known hotspots
        features["hotspot_flag"] = int(pos in hotspot_positions)

        # 18. CpG Overlap (is there CG dinucleotide in window?)
        features["cpg_overlap"] = int('CG' in window_seq)

        # ---- Return
        debug_info = (window_seq, mut_window, gc_ref, gc_content(mut_window), round(gc_content(mut_window) - gc_ref, 3), align_score)
        return pd.DataFrame([features])[self.expected_features], debug_info

    def setup_lime_explainer(self, training_data: Optional[pd.DataFrame] = None):
        """Set up LIME explainer with synthetic training data if real data not available"""
        if training_data is not None:
            self.training_data = training_data
        else:
            # Generate synthetic training data for LIME
            np.random.seed(42)
            n_samples = 1000
            
            synthetic_data = []
            for _ in range(n_samples):
                # Generate realistic synthetic features
                sample = {feature: 0 for feature in self.expected_features}
                
                # Position-based features
                sample["position"] = np.random.randint(154021500, 154022200)
                sample["splice_distance"] = np.random.randint(0, 500)
                sample["position_bin"] = np.random.randint(1, 11)
                sample["position_decile"] = sample["position_bin"]
                sample["chr_chrX"] = 1
                
                # Binary features
                sample["region_exon"] = np.random.choice([0, 1])
                sample["region_non-exon"] = 1 - sample["region_exon"]
                sample["type"] = np.random.randint(0, 4)
                
                # Molecular consequences (only one typically active)
                mc_choice = np.random.choice(list(self.mc_labels.values()))
                sample[mc_choice] = 1
                
                # Nucleotide context
                for base in ['A', 'C', 'G', 'T']:
                    sample[f"prev_{base}"] = np.random.choice([0, 1], p=[0.7, 0.3])
                    sample[f"next_{base}"] = np.random.choice([0, 1], p=[0.7, 0.3])
                
                # Continuous features
                sample["alignment_score"] = np.random.randint(15, 21)
                sample["gc_content"] = np.random.uniform(0.3, 0.7)
                
                synthetic_data.append(sample)
            
            self.training_data = pd.DataFrame(synthetic_data)[self.expected_features]
        
        # Set up categorical features (binary and categorical columns)
        categorical_features = []
        for i, feature in enumerate(self.expected_features):
            if feature.startswith('mc_') or feature.startswith('prev_') or feature.startswith('next_') or \
               feature in ['region_exon', 'region_non-exon', 'chr_chrX', 'type']:
                categorical_features.append(i)
        
        self.lime_explainer = LimeTabularExplainer(
            self.training_data.values,
            feature_names=self.expected_features,
            categorical_features=categorical_features,
            class_names=['Benign', 'Pathogenic'],
            mode='classification'
        )
        
        logger.info(f"LIME explainer initialized with {len(self.training_data)} training samples")

    def explain_prediction(self, features_df: pd.DataFrame, num_features: int = 10) -> dict:
        """Generate LIME explanation for a prediction"""
        if self.lime_explainer is None:
            self.setup_lime_explainer()
        
        # Get the instance to explain
        instance = features_df.iloc[0].values
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            instance, 
            self.model.predict_proba, 
            num_features=num_features,
            num_samples=500
        )
        
        # Extract explanation data
        exp_list = explanation.as_list()
        exp_map = explanation.as_map()
        
        # Process explanation
        feature_importance = []
        for feature_name, importance in exp_list:
            feature_importance.append({
                'feature': feature_name,
                'importance': importance,
                'value': features_df.iloc[0][feature_name] if feature_name in features_df.columns else 'N/A'
            })
        
        return {
            'explanation': explanation,
            'feature_importance': feature_importance,
            'exp_map': exp_map
        }

    def predict(self, spdi: str, consequences: List[str]) -> Tuple[int, float, pd.DataFrame, Tuple]:
        chrom, pos, deleted, inserted = spdi.strip().split(":")
        pos = int(pos)
        df, debug = self.compute_features(pos, deleted, inserted, consequences)
        
        # 🔒 Enforce frameshift rule
        if "mc_frameshift_variant" in df.columns and df["mc_frameshift_variant"].iloc[0] == 1:
            return 1, 1.0, df, debug  # Force as Pathogenic with 100% confidence
        
        pred = self.model.predict(df)[0]
        try:
            prob = self.model.predict_proba(df)[0][1]
        except:
            prob = 0.5
        return int(pred), float(prob), df, debug

def create_lime_plot(feature_importance: List[dict], prediction: int) -> plt.Figure:
    """Create a horizontal bar plot for LIME feature importance"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by absolute importance
    sorted_features = sorted(feature_importance, key=lambda x: abs(x['importance']), reverse=True)
    
    features = [f"{item['feature']}" for item in sorted_features]
    importances = [item['importance'] for item in sorted_features]
    
    # Color coding: positive importance = red (towards pathogenic), negative = green (towards benign)
    colors = ['red' if imp > 0 else 'green' for imp in importances]
    
    bars = ax.barh(features, importances, color=colors, alpha=0.7)
    
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'LIME Explanation - Prediction: {"Pathogenic" if prediction else "Benign"}')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars, importances):
        width = bar.get_width()
        ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

# ---- STREAMLIT APP ----
def main():
    st.set_page_config(page_title="MECP2 Classifier with LIME", layout="wide")
    st.title("🧬 MECP2 Mutation Pathogenicity Classifier with Explainability")
    st.markdown("*Predict pathogenicity of MECP2 mutations with LIME explanations*")

    model_path = "models/best_model_random_forest.pkl"
    gtf_path = "mecp2_exons.tsv"

    try:
        classifier = MECP2Classifier(model_path, gtf_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.info("Please ensure the model file and GTF file are available.")
        return

    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        spdi = st.text_input("Enter Canonical SPDI:", value="NC_000023.11:154021866:G:C")
        consequence_input = st.multiselect(
            "Select Molecular Consequences:", 
            list(classifier.mc_labels.keys()), 
            default=["missense variant"]
        )
    
    with col2:
        st.markdown("### SPDI Format")
        st.markdown("""
        - **Sequence**: NC_000023.11 (chrX)
        - **Position**: genomic position
        - **Deletion**: reference allele
        - **Insertion**: alternative allele
        """)

    # Prediction section
    if st.button("🔬 Predict & Explain", type="primary"):
        if not spdi or not consequence_input:
            st.error("Please provide both SPDI and molecular consequences.")
            return
        
        try:
            with st.spinner("Generating prediction and explanation..."):
                pred, prob, df, debug = classifier.predict(spdi, consequence_input)
                
                # Display main results
                st.success("✅ Prediction completed!")
                
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    prediction_text = '🔴 Pathogenic' if pred else '🟢 Benign'
                    st.metric("Prediction", prediction_text)
                with result_col2:
                    st.metric("Confidence Score", f"{prob:.3f}")
                
                # Generate LIME explanation
                st.subheader("🎯 LIME Explainability Analysis")
                
                with st.spinner("Generating LIME explanation..."):
                    lime_results = classifier.explain_prediction(df, num_features=15)
                
                # Display LIME results
                tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "📋 Detailed Features", "🔬 Sequence Analysis"])
                
                with tab1:
                    st.markdown("**Top factors influencing this prediction:**")
                    
                    # Create and display LIME plot
                    fig = create_lime_plot(lime_results['feature_importance'], pred)
                    st.pyplot(fig)
                    
                    # Summary text
                    top_features = lime_results['feature_importance'][:5]
                    st.markdown("**Key insights:**")
                    for i, feat in enumerate(top_features, 1):
                        direction = "towards Pathogenic" if feat['importance'] > 0 else "towards Benign"
                        st.markdown(f"{i}. **{feat['feature']}** (value: {feat['value']}) - "
                                  f"Importance: {feat['importance']:.3f} ({direction})")
                
                with tab2:
                    st.markdown("**All computed features:**")
                    
                    # Display feature importance table
                    importance_df = pd.DataFrame(lime_results['feature_importance'])
                    importance_df = importance_df.sort_values('importance', key=abs, ascending=False)
                    
                    st.dataframe(
                        importance_df,
                        column_config={
                            "feature": "Feature Name",
                            "importance": st.column_config.NumberColumn("LIME Importance", format="%.4f"),
                            "value": "Feature Value"
                        },
                        hide_index=True
                    )
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download Feature Importance", 
                            data=importance_df.to_csv(index=False).encode('utf-8'), 
                            file_name="lime_feature_importance.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.download_button(
                            "📥 Download All Features", 
                            data=df.to_csv(index=False).encode('utf-8'), 
                            file_name="all_features.csv",
                            mime="text/csv"
                        )
                
                with tab3:
                    # Show sequence window analysis
                    ref_win, mut_win, gc_orig, gc_var, gc_delta, align_score = debug
                    
                    st.markdown("**Sequence Context Analysis:**")
                    
                    seq_col1, seq_col2 = st.columns(2)
                    with seq_col1:
                        st.text_area("Reference Sequence", ref_win, height=100)
                    with seq_col2:
                        st.text_area("Mutated Sequence", mut_win, height=100)
                    
                    # Sequence metrics
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    metric_col1.metric("GC Content (Ref)", f"{gc_orig:.3f}")
                    metric_col2.metric("GC Content (Mut)", f"{gc_var:.3f}")
                    metric_col3.metric("GC Δ", f"{gc_delta:+.3f}")
                    metric_col4.metric("Alignment Score", f"{align_score}")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please check your input format and try again.")

    # Information sidebar
    with st.sidebar:
        st.markdown("### ℹ️ About This Tool")
        st.markdown("""
        This tool predicts the pathogenicity of MECP2 mutations using machine learning
        and provides explanations using LIME (Local Interpretable Model-agnostic Explanations).
        
        **Features:**
        - Pathogenicity prediction
        - LIME explainability analysis
        - Sequence context analysis
        - Feature importance ranking
        
        **Input Requirements:**
        - SPDI format for mutations
        - Molecular consequences
        """)
        
        st.markdown("### 🎯 LIME Explanation")
        st.markdown("""
        LIME explains individual predictions by:
        - Showing which features most influenced the decision
        - Indicating whether features push towards pathogenic (red) or benign (green)
        - Providing local explanations for each specific case
        """)

if __name__ == "__main__":
    main()