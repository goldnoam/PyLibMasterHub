
import { Library, Category } from './types';

export const CATEGORIES: Category[] = [
  'Agentic AI',
  'Generative AI',
  'Data Manipulation',
  'Database Operation',
  'Machine Learning',
  'Data Visualization',
  'Time Series Analysis',
  'Natural Language Processing',
  'Statistical Analysis',
  'Web Scraping'
];

export const LIBRARIES: Library[] = [
  // Agentic AI
  { id: 'langchain', name: 'LangChain', category: 'Agentic AI', icon: '⛓️', shortDescription: 'Framework for building LLM applications.', officialUrl: 'https://python.langchain.com/' },
  { id: 'autogpt', name: 'AutoGPT', category: 'Agentic AI', icon: '🤖', shortDescription: 'An experimental open-source AI agent.', officialUrl: 'https://github.com/Significant-Gravitas/AutoGPT' },
  { id: 'haystack', name: 'Haystack', category: 'Agentic AI', icon: '🌾', shortDescription: 'Orchestration for search and RAG.', officialUrl: 'https://haystack.deepset.ai/' },
  
  // Generative AI
  { id: 'huggingface', name: 'Hugging Face', category: 'Generative AI', icon: '🤗', shortDescription: 'The hub for pre-trained models.', officialUrl: 'https://huggingface.co/' },
  { id: 'openai', name: 'OpenAI', category: 'Generative AI', icon: '🦾', shortDescription: 'The state-of-the-art AI interface.', officialUrl: 'https://platform.openai.com/' },
  { id: 'diffusers', name: 'Diffusers', category: 'Generative AI', icon: '🧨', shortDescription: 'State-of-the-art diffusion models for image generation.', officialUrl: 'https://github.com/huggingface/diffusers' },
  
  // Data Manipulation
  { id: 'numpy', name: 'NumPy', category: 'Data Manipulation', icon: '🔢', shortDescription: 'Fundamental package for scientific computing.', officialUrl: 'https://numpy.org/' },
  { id: 'pandas', name: 'Pandas', category: 'Data Manipulation', icon: '🐼', shortDescription: 'Data analysis and manipulation library.', officialUrl: 'https://pandas.pydata.org/' },
  { id: 'polars', name: 'Polars', category: 'Data Manipulation', icon: '❄️', shortDescription: 'Blazingly fast DataFrame library in Rust.', officialUrl: 'https://www.pola.rs/' },
  
  // ML
  { id: 'scikitlearn', name: 'Scikit-Learn', category: 'Machine Learning', icon: '🧠', shortDescription: 'Simple and efficient tools for predictive data analysis.', officialUrl: 'https://scikit-learn.org/' },
  { id: 'tensorflow', name: 'TensorFlow', category: 'Machine Learning', icon: '🟧', shortDescription: 'End-to-end open source platform for ML.', officialUrl: 'https://www.tensorflow.org/' },
  { id: 'pytorch', name: 'PyTorch', category: 'Machine Learning', icon: '🔥', shortDescription: 'Deep learning framework with dynamic graphs.', officialUrl: 'https://pytorch.org/' },
  
  // Visualization
  { id: 'matplotlib', name: 'Matplotlib', category: 'Data Visualization', icon: '📊', shortDescription: 'Comprehensive library for creating static visualizations.', officialUrl: 'https://matplotlib.org/' },
  { id: 'seaborn', name: 'Seaborn', category: 'Data Visualization', icon: '🌊', shortDescription: 'Statistical data visualization built on Matplotlib.', officialUrl: 'https://seaborn.pydata.org/' },
  { id: 'plotly', name: 'Plotly', category: 'Data Visualization', icon: '📈', shortDescription: 'Interactive, publication-quality graphs.', officialUrl: 'https://plotly.com/python/' },
  
  // NLP
  { id: 'nltk', name: 'NLTK', category: 'Natural Language Processing', icon: '📜', shortDescription: 'The Natural Language Toolkit.', officialUrl: 'https://www.nltk.org/' },
  { id: 'spacy', name: 'spaCy', category: 'Natural Language Processing', icon: '🪐', shortDescription: 'Industrial-strength Natural Language Processing.', officialUrl: 'https://spacy.io/' },
  
  // Web Scraping
  { id: 'beautifulsoup', name: 'Beautiful Soup', category: 'Web Scraping', icon: '🍲', shortDescription: 'Parsing HTML and XML documents.', officialUrl: 'https://www.crummy.com/software/BeautifulSoup/' },
  { id: 'scrapy', name: 'Scrapy', category: 'Web Scraping', icon: '🕷️', shortDescription: 'A fast high-level web crawling and scraping framework.', officialUrl: 'https://scrapy.org/' },
  { id: 'selenium', name: 'Selenium', category: 'Web Scraping', icon: '✅', shortDescription: 'Automating web browsers.', officialUrl: 'https://www.selenium.dev/' }
];
