import streamlit as st
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.multimodal_model import MultimodalContentGenerator
from src.data_processing import DataManager
import time

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Content Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
}
.result-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the multimodal model (cached for performance)"""
    return MultimodalContentGenerator()

@st.cache_data
def load_data_manager():
    """Load data manager (cached for performance)"""
    return DataManager()

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 Multimodal AI Content Generator</h1>', unsafe_allow_html=True)
    st.markdown("### Transform images and text into rich, dynamic content using cutting-edge AI")
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Load components
    with st.spinner("🔄 Loading AI models..."):
        try:
            model = load_model()
            data_manager = load_data_manager()
            st.sidebar.success("✅ Models loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading models: {e}")
            st.stop()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Image Analysis", 
        "📝 Caption Generation", 
        "🎭 Mood Enhancement", 
        "🔊 Soundscape Suggestions"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Image-Text Similarity Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload or Select Image")
            
            # Image input options
            image_source = st.radio(
                "Choose image source:",
                ["Upload Image", "Use Sample Images", "URL"]
            )
            
            image_path = None
            
            if image_source == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
                )
                if uploaded_file:
                    # Save uploaded file
                    image_path = f"data/images/{uploaded_file.name}"
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            elif image_source == "Use Sample Images":
                # Prepare sample data if needed
                if st.button("📥 Download Sample Images"):
                    with st.spinner("Downloading sample images..."):
                        data_manager.prepare_demo_data()
                    st.success("Sample images ready!")
                
                available_images = data_manager.get_available_images()
                if available_images:
                    selected_image = st.selectbox("Select sample image:", available_images)
                    if selected_image:
                        image_path = selected_image
                        st.image(selected_image, caption="Selected Image", use_column_width=True)
                else:
                    st.info("Click 'Download Sample Images' to get started!")
            
            elif image_source == "URL":
                image_url = st.text_input("Enter image URL:")
                if image_url:
                    try:
                        st.image(image_url, caption="Image from URL", use_column_width=True)
                        image_path = image_url
                    except:
                        st.error("Invalid image URL")
        
        with col2:
            st.subheader("📝 Text Descriptions")
            
            # Text input
            text_descriptions = st.text_area(
                "Enter text descriptions (one per line):",
                "a beautiful nature scene\na cute domestic animal\na busy city street\nan artistic masterpiece",
                height=150
            )
            
            if st.button("🔍 Analyze Similarity", type="primary"):
                if image_path and text_descriptions:
                    descriptions_list = [desc.strip() for desc in text_descriptions.split('\n') if desc.strip()]
                    
                    with st.spinner("Analyzing image-text similarity..."):
                        similarities = model.analyze_image_text_similarity(image_path, descriptions_list)
                    
                    if similarities is not None:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.subheader("📊 Similarity Results")
                        
                        # Create results dataframe for better display
                        import pandas as pd
                        results_df = pd.DataFrame({
                            'Description': descriptions_list,
                            'Similarity Score': similarities,
                            'Confidence': [f"{score:.1%}" for score in similarities]
                        })
                        results_df = results_df.sort_values('Similarity Score', ascending=False)
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Highlight best match
                        best_match_idx = similarities.argmax()
                        st.success(f"🎯 Best match: '{descriptions_list[best_match_idx]}' (Score: {similarities[best_match_idx]:.3f})")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("Failed to analyze similarity")
                else:
                    st.warning("Please provide both an image and text descriptions")
    
    with tab2:
        st.markdown('<h2 class="sub-header">AI-Powered Image Captioning</h2>', unsafe_allow_html=True)
        
        if image_path:
            if st.button("✨ Generate Caption", type="primary"):
                with st.spinner("Generating detailed caption..."):
                    caption = model.generate_image_caption(image_path)
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("📖 Generated Caption")
                st.write(f"**{caption}**")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Select an image in the Image Analysis tab first")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Mood-Adaptive Content Enhancement</h2>', unsafe_allow_html=True)
        
        base_description = st.text_input(
            "Base description:",
            "A serene landscape with mountains and water"
        )
        
        target_mood = st.selectbox(
            "Target emotional mood:",
            ["upbeat", "melancholy", "mysterious", "peaceful", "adventurous"]
        )
        
        if st.button("🎭 Enhance with Mood", type="primary"):
            enhanced = model.suggest_mood_enhancements(base_description, target_mood)
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("✨ Mood-Enhanced Description")
            st.write(f"**Original:** {base_description}")
            st.write(f"**Enhanced ({target_mood}):** {enhanced}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Soundscape Suggestions</h2>', unsafe_allow_html=True)
        
        if image_path:
            selected_mood = st.selectbox(
                "Audio mood:",
                ["peaceful", "energetic", "nature", "urban"]
            )
            
            if st.button("🔊 Generate Soundscape", type="primary"):
                with st.spinner("Generating soundscape suggestions..."):
                    # Get image caption first
                    caption = model.generate_image_caption(image_path)
                    # Generate soundscape suggestions
                    suggestions = model.generate_soundscape_suggestions(caption, selected_mood)
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("🎵 Recommended Soundscape")
                st.write(f"**Based on:** {caption}")
                st.write("**Suggested audio elements:**")
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"{i}. {suggestion}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Select an image in the Image Analysis tab first")
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 Built with:** Streamlit, CLIP, BLIP, PyTorch | **👨‍💻 Developer:** Achutha Gowda")

if __name__ == "__main__":
    main()
