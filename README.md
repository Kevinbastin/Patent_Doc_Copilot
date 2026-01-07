# 📝 PatentDoc Co-Pilot

**Enterprise-Level IPO-Compliant Patent Drafting System**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **One-Click Generation** - Generate all 9 patent sections instantly
- **IPO Compliance** - Follows Indian Patent Office format exactly
- **Multi-Agent Verification** - 5 AI agents verify document quality
- **Prior Art Search** - Automated patent database searching
- **Perfect Diagrams** - Generate block diagrams, flowcharts, sequences
- **Complete Filing Package** - Forms 1, 3, 5 included
- **Professional Export** - PDF & DOCX with embedded diagrams

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/Patent_Doc_Copilot.git
cd Patent_Doc_Copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
streamlit run app.py
```

## 🔑 API Keys Required

Create a `.env` file with:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
HUGGINGFACE_API_KEY=your_hf_key_here  # Optional, for images
```

## 📋 IPO Sections Generated

1. Title of the Invention
2. Field of the Invention
3. Background of the Invention
4. Objects of the Invention
5. Summary of the Invention
6. Brief Description of Drawings
7. Detailed Description
8. Industrial Applicability
9. Claims

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets (API keys)
5. Deploy!

### Docker

```bash
docker build -t patentdoc-copilot .
docker run -p 8501:8501 -e OPENROUTER_API_KEY=your_key patentdoc-copilot
```

## 📄 License

MIT License - Free for commercial and personal use.

## ⚠️ Disclaimer

This tool assists with patent drafting but is not a substitute for professional legal advice. Always consult a qualified patent attorney before filing.

---

Made with ❤️ for inventors and innovators