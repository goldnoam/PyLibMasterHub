
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
  { 
    id: 'numpy', 
    name: 'NumPy', 
    category: 'Data Manipulation', 
    icon: '🔢', 
    shortDescription: 'The fundamental package for scientific computing with Python.',
    longDescription: 'NumPy is the backbone of the Python scientific stack. It provides a powerful N-dimensional array object, sophisticated broadcasting functions, and tools for integrating C/C++ and Fortran code.',
    keyFeatures: ['N-dimensional array object', 'Broadcasting functions', 'Linear algebra & Fourier transform', 'Random number capabilities'],
    codeExample: 'import numpy as np\n\n# Create a 2D array\na = np.array([[1, 2], [3, 4]])\nb = np.array([[5, 6], [7, 8]])\n\n# Element-wise multiplication\nresult = a * b\nprint(f"Matrix Result:\\n{result}")',
    officialUrl: 'https://numpy.org/' 
  },
  { 
    id: 'pandas', 
    name: 'Pandas', 
    category: 'Data Manipulation', 
    icon: '🐼', 
    shortDescription: 'Fast, powerful, and flexible data analysis and manipulation tool.',
    longDescription: 'Pandas is a high-performance open-source library providing easy-to-use data structures and data analysis tools for the Python programming language.',
    keyFeatures: ['DataFrame object for data manipulation', 'Tools for reading/writing data', 'Handling of missing data', 'Dataset merging and joining'],
    codeExample: 'import pandas as pd\n\n# Create a simple DataFrame\ndata = {\n    "Name": ["Alice", "Bob", "Charlie"],\n    "Age": [25, 30, 35]\n}\ndf = pd.DataFrame(data)\n\n# Filter rows\nprint(df[df["Age"] > 28])',
    officialUrl: 'https://pandas.pydata.org/' 
  },
  { 
    id: 'tensorflow', 
    name: 'TensorFlow', 
    category: 'Machine Learning', 
    icon: '🟧', 
    shortDescription: 'An end-to-end open source platform for machine learning.',
    longDescription: 'Developed by the Google Brain team, TensorFlow is a comprehensive ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML.',
    keyFeatures: ['Scalable neural network training', 'Support for multiple platforms (Mobile, Edge, Cloud)', 'Robust ecosystem (TensorBoard, TF Hub)', 'Eager execution for debugging'],
    codeExample: 'import tensorflow as tf\n\n# Simple linear model\nmodel = tf.keras.Sequential([\n    tf.keras.layers.Dense(units=1, input_shape=[1])\n])\n\nmodel.compile(optimizer="sgd", loss="mean_squared_error")\nprint("Model summary:", model.summary())',
    officialUrl: 'https://www.tensorflow.org/' 
  },
  { 
    id: 'pytorch', 
    name: 'PyTorch', 
    category: 'Machine Learning', 
    icon: '🔥', 
    shortDescription: 'Tensors and Dynamic neural networks in Python with strong GPU acceleration.',
    longDescription: 'PyTorch is an optimized tensor library for deep learning using GPUs and CPUs. It is favored by researchers for its dynamic computation graph and Pythonic nature.',
    keyFeatures: ['Dynamic Computation Graphs', 'Strong GPU acceleration', 'TorchScript for production', 'Native support for distributed training'],
    codeExample: 'import torch\nimport torch.nn as nn\n\n# Create a tensor\nx = torch.randn(3, 3)\ny = torch.randn(3, 3)\n\nresult = x + y\nprint(result)',
    officialUrl: 'https://pytorch.org/' 
  },
  { 
    id: 'matplotlib', 
    name: 'Matplotlib', 
    category: 'Data Visualization', 
    icon: '📊', 
    shortDescription: 'Comprehensive library for creating static, animated, and interactive visualizations.',
    longDescription: 'Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications.',
    keyFeatures: ['Line plots, scatter plots, bar charts', 'Histogram and error charts', '3D plotting support', 'Export to PDF, SVG, PNG, and more'],
    codeExample: 'import matplotlib.pyplot as plt\n\nplt.plot([1, 2, 3, 4], [1, 4, 9, 16])\nplt.ylabel("some numbers")\nplt.show()',
    officialUrl: 'https://matplotlib.org/' 
  },
  { 
    id: 'langchain', 
    name: 'LangChain', 
    category: 'Agentic AI', 
    icon: '⛓️', 
    shortDescription: 'Framework for building applications powered by language models.',
    longDescription: 'LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.',
    keyFeatures: ['Prompt Management', 'Chains of LLM calls', 'Data augmented generation (RAG)', 'Memory and state management'],
    codeExample: 'from langchain.llms import OpenAI\nfrom langchain.prompts import PromptTemplate\n\ntemplate = "What is a good name for a company that makes {product}?"\nprompt = PromptTemplate(input_variables=["product"], template=template)',
    officialUrl: 'https://python.langchain.com/' 
  },
  { 
    id: 'scikit-learn', 
    name: 'Scikit-learn', 
    category: 'Machine Learning', 
    icon: '🤖', 
    shortDescription: 'Simple and efficient tools for predictive data analysis.',
    longDescription: 'Scikit-learn is a premier library for classical machine learning, built on NumPy, SciPy, and matplotlib. It features various classification, regression and clustering algorithms.',
    keyFeatures: ['Classification (SVM, Random Forest)', 'Regression (Linear, Ridge)', 'Clustering (K-Means)', 'Dimensionality Reduction (PCA)'],
    codeExample: 'from sklearn.ensemble import RandomForestClassifier\n\n# Sample data\nX = [[0, 0], [1, 1]]\ny = [0, 1]\n\nclf = RandomForestClassifier(n_estimators=10)\nclf = clf.fit(X, y)\nprint(clf.predict([[2, 2]]))',
    officialUrl: 'https://scikit-learn.org/' 
  },
  { 
    id: 'seaborn', 
    name: 'Seaborn', 
    category: 'Data Visualization', 
    icon: '🌊', 
    shortDescription: 'Statistical data visualization based on matplotlib.',
    longDescription: 'Seaborn provides a high-level interface for drawing attractive and informative statistical graphics. It integrates closely with pandas data structures.',
    keyFeatures: ['Dataset-oriented API', 'Integrated regression estimation', 'Complex multi-plot grids', 'Beautiful default themes'],
    codeExample: 'import seaborn as sns\nimport matplotlib.pyplot as plt\n\n# Load example dataset\ntips = sns.load_dataset("tips")\n\n# Create visualization\nsns.relplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker")\nplt.show()',
    officialUrl: 'https://seaborn.pydata.org/' 
  },
  { 
    id: 'beautifulsoup', 
    name: 'BeautifulSoup', 
    category: 'Web Scraping', 
    icon: '🥣', 
    shortDescription: 'Library for pulling data out of HTML and XML files.',
    longDescription: 'Beautiful Soup provides simple methods for navigating, searching, and modifying a parse tree. It commonly uses the requests library to fetch web pages.',
    keyFeatures: ['Navigating the parse tree', 'Searching with filters', 'Fixing bad HTML', 'Output formatting'],
    codeExample: 'from bs4 import BeautifulSoup\n\nhtml_doc = "<html><head><title>The Dormouse\'s story</title></head>"\nsoup = BeautifulSoup(html_doc, "html.parser")\n\nprint(soup.title.string)\n# Output: The Dormouse\'s story',
    officialUrl: 'https://www.crummy.com/software/BeautifulSoup/' 
  },
  { 
    id: 'sqlalchemy', 
    name: 'SQLAlchemy', 
    category: 'Database Operation', 
    icon: '🏛️', 
    shortDescription: 'The Python SQL Toolkit and Object Relational Mapper.',
    longDescription: 'SQLAlchemy is the ultimate database access tool for Python. It provides a full suite of well-known enterprise-level persistence patterns.',
    keyFeatures: ['Object Relational Mapper (ORM)', 'SQL Expression Language', 'Database Abstraction Layer', 'Connection Pooling'],
    codeExample: 'from sqlalchemy import create_engine, Column, Integer, String\nfrom sqlalchemy.orm import declarative_base\n\nBase = declarative_base()\nclass User(Base):\n    __tablename__ = "users"\n    id = Column(Integer, primary_key=True)\n    name = Column(String)\n\nengine = create_engine("sqlite:///:memory:")\nBase.metadata.create_all(engine)',
    officialUrl: 'https://www.sqlalchemy.org/' 
  },
  { 
    id: 'scipy', 
    name: 'SciPy', 
    category: 'Statistical Analysis', 
    icon: '🔬', 
    shortDescription: 'Fundamental library for scientific computing and advanced mathematics.',
    longDescription: 'SciPy contains modules for optimization, linear algebra, integration, interpolation, special functions, FFT, and signal/image processing.',
    keyFeatures: ['Numerical Integration', 'Optimization algorithms', 'Signal processing tools', 'Statistical distributions'],
    codeExample: 'from scipy import optimize\n\ndef f(x):\n    return x**2 + 10*np.sin(x)\n\nresult = optimize.minimize(f, x0=0)\nprint(f"Minimum found at: {result.x}")',
    officialUrl: 'https://scipy.org/' 
  },
  { 
    id: 'spacy', 
    name: 'spaCy', 
    category: 'Natural Language Processing', 
    icon: '🚀', 
    shortDescription: 'Industrial-strength Natural Language Processing in Python.',
    longDescription: 'spaCy is designed specifically for production use. It helps you build applications that process and "understand" large volumes of text.',
    keyFeatures: ['Tokenization & Lemmatization', 'Entity Recognition', 'Part-of-speech tagging', 'Pre-trained neural models'],
    codeExample: 'import spacy\n\nnlp = spacy.load("en_core_web_sm")\ndoc = nlp("Apple is looking at buying U.K. startup for $1 billion")\n\nfor ent in doc.ents:\n    print(ent.text, ent.label_)',
    officialUrl: 'https://spacy.io/' 
  },
  { 
    id: 'transformers', 
    name: 'Transformers', 
    category: 'Generative AI', 
    icon: '🤗', 
    shortDescription: 'State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.',
    longDescription: 'Developed by Hugging Face, this library provides thousands of pre-trained models to perform tasks on texts such as classification, extraction, and generation.',
    keyFeatures: ['Large Language Models (LLMs)', 'Image Classification', 'Audio processing', 'Multi-modal capabilities'],
    codeExample: 'from transformers import pipeline\n\nclassifier = pipeline("sentiment-analysis")\nresult = classifier("I love using Python libraries!")\nprint(result)',
    officialUrl: 'https://huggingface.co/docs/transformers/index' 
  },
  { 
    id: 'plotly', 
    name: 'Plotly', 
    category: 'Data Visualization', 
    icon: '📈', 
    shortDescription: 'Interactive, browser-based graphing library.',
    longDescription: 'Plotly is a technical computing library that allows you to create beautiful interactive charts. It is excellent for dashboards and data apps.',
    keyFeatures: ['Hover interactions', 'Zooming & Panning', 'Export to HTML', 'Financial and Scientific charts'],
    codeExample: 'import plotly.express as px\n\ndf = px.data.iris()\nfig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")\nfig.show()',
    officialUrl: 'https://plotly.com/python/' 
  },
  { 
    id: 'prophet', 
    name: 'Prophet', 
    category: 'Time Series Analysis', 
    icon: '🔮', 
    shortDescription: 'Forecasting procedure for time series data.',
    longDescription: 'Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality.',
    keyFeatures: ['Handles missing data', 'Robust to outliers', 'Automatic seasonality', 'Holiday effects inclusion'],
    codeExample: 'from prophet import Prophet\nimport pandas as pd\n\ndf = pd.read_csv("example_wp_log_peyton_manning.csv")\nm = Prophet()\nm.fit(df)\n\nfuture = m.make_future_dataframe(periods=365)\nforecast = m.predict(future)',
    officialUrl: 'https://facebook.github.io/prophet/' 
  },
  { 
    id: 'statsmodels', 
    name: 'Statsmodels', 
    category: 'Statistical Analysis', 
    icon: '📐', 
    shortDescription: 'Statistical modeling and testing in Python.',
    longDescription: 'Statsmodels is a Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests.',
    keyFeatures: ['Linear Regression (OLS)', 'Generalized Linear Models', 'Time series analysis (ARIMA)', 'Nonparametric methods'],
    codeExample: 'import statsmodels.api as sm\n\nX = sm.add_constant([1, 2, 3, 4, 5])\ny = [1, 3, 4, 5, 2]\n\nmodel = sm.OLS(y, X).fit()\nprint(model.summary())',
    officialUrl: 'https://www.statsmodels.org/' 
  },
  { 
    id: 'crewai', 
    name: 'CrewAI', 
    category: 'Agentic AI', 
    icon: '👥', 
    shortDescription: 'Framework for orchestrating role-playing, autonomous AI agents.',
    longDescription: 'CrewAI allows you to create collaborative teams of AI agents that work together to solve complex tasks. It focuses on agent delegation and process management.',
    keyFeatures: ['Role-based agent design', 'Task delegation', 'Customizable workflows', 'Integration with LangChain tools'],
    codeExample: 'from crewai import Agent, Task, Crew\n\nresearcher = Agent(role="Researcher", goal="Analyze AI trends", backstory="Expert analyst")\ntask = Task(description="Summarize 2024 AI news", agent=researcher)\n\ncrew = Crew(agents=[researcher], tasks=[task])\nresult = crew.kickoff()',
    officialUrl: 'https://www.crewai.com/' 
  },
  { 
    id: 'polars', 
    name: 'Polars', 
    category: 'Data Manipulation', 
    icon: '🧊', 
    shortDescription: 'Blazingly fast DataFrames library written in Rust.',
    longDescription: 'Polars is a highly performant DataFrame library for manipulating structured data. It leverages Apache Arrow and parallel execution for maximum speed.',
    keyFeatures: ['Parallel execution', 'Lazy evaluation', 'Memory efficient', 'Rust-based performance'],
    codeExample: 'import polars as pl\n\ndf = pl.DataFrame({\n    "foo": [1, 2, 3],\n    "bar": ["a", "b", "c"]\n})\n\nprint(df.filter(pl.col("foo") > 1))',
    officialUrl: 'https://www.pola.rs/' 
  },
  { 
    id: 'scrapy', 
    name: 'Scrapy', 
    category: 'Web Scraping', 
    icon: '🕸️', 
    shortDescription: 'Fast high-level web crawling and scraping framework.',
    longDescription: 'Scrapy is an open source and collaborative framework for extracting the data you need from websites in a clean, fast way.',
    keyFeatures: ['Built-in CSS/XPath selectors', 'Async request handling', 'Robust spider middleware', 'Data export to JSON, CSV'],
    codeExample: 'import scrapy\n\nclass BlogSpider(scrapy.Spider):\n    name = "blogspider"\n    start_urls = ["https://blog.scrapinghub.com"]\n\n    def parse(self, response):\n        for title in response.css(".post-header h2 a::text"):\n            yield {"title": title.get()}',
    officialUrl: 'https://scrapy.org/' 
  },
  { 
    id: 'nltk', 
    name: 'NLTK', 
    category: 'Natural Language Processing', 
    icon: '📜', 
    shortDescription: 'The Natural Language Toolkit for symbolic and statistical NLP.',
    longDescription: 'NLTK is a leading platform for building Python programs to work with human language data. It provides interfaces to over 50 corpora and lexical resources.',
    keyFeatures: ['Text classification', 'Tokenization & Tagging', 'Parsing & Semantic reasoning', 'Large collection of corpora'],
    codeExample: 'import nltk\nfrom nltk.tokenize import word_tokenize\n\ntext = "NLTK is great for text processing."\ntokens = word_tokenize(text)\nprint(nltk.pos_tag(tokens))',
    officialUrl: 'https://www.nltk.org/' 
  },
  { 
    id: 'peewee', 
    name: 'Peewee', 
    category: 'Database Operation', 
    icon: '🐕', 
    shortDescription: 'A small, expressive ORM for SQLite, MySQL, and PostgreSQL.',
    longDescription: 'Peewee is a simple and small ORM. It has few concepts, making it easy to learn and intuitive to use. Perfect for small to medium-sized projects.',
    keyFeatures: ['Expressive query syntax', 'Lightweight & Small footprint', 'Built-in support for multiple DBs', 'Easy model definitions'],
    codeExample: 'from peewee import *\n\ndb = SqliteDatabase("people.db")\nclass Person(Model):\n    name = CharField()\n    class Meta:\n        database = db\n\ndb.connect()\ndb.create_tables([Person])',
    officialUrl: 'http://docs.peewee-orm.com/' 
  },
  { 
    id: 'xgboost', 
    name: 'XGBoost', 
    category: 'Machine Learning', 
    icon: '🏹', 
    shortDescription: 'Scalable and flexible gradient boosting library.',
    longDescription: 'XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.',
    keyFeatures: ['Gradient Tree Boosting', 'Parallel & Distributed computing', 'Cross-platform support', 'Feature importance ranking'],
    codeExample: 'import xgboost as xgb\nfrom sklearn.datasets import load_iris\n\ndata, target = load_iris(return_X_y=True)\ndtrain = xgb.DMatrix(data, label=target)\n\nparam = {"max_depth": 2, "eta": 1, "objective": "multi:softprob", "num_class": 3}\nbst = xgb.train(param, dtrain, num_boost_round=2)',
    officialUrl: 'https://xgboost.readthedocs.io/' 
  }
];
