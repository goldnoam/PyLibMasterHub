
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
    codeExample: 'import torch\nimport torch.nn as nn\n\n# Create a tensor\nx = torch.randn(3, 3)\ny = torch.randn(3, 3)\n\n# Addition with GPU if available\nif torch.cuda.is_available():\n    x, y = x.cuda(), y.cuda()\n\nresult = x + y\nprint(result)',
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
    codeExample: 'from langchain.llms import OpenAI\nfrom langchain.prompts import PromptTemplate\n\n# This is a conceptual example\ntemplate = "What is a good name for a company that makes {product}?"\nprompt = PromptTemplate(input_variables=["product"], template=template)',
    officialUrl: 'https://python.langchain.com/' 
  }
];
