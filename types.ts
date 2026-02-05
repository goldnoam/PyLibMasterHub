
export type Category = 
  | 'Agentic AI'
  | 'Generative AI'
  | 'Data Manipulation'
  | 'Database Operation'
  | 'Machine Learning'
  | 'Data Visualization'
  | 'Time Series Analysis'
  | 'Natural Language Processing'
  | 'Statistical Analysis'
  | 'Web Scraping';

export interface Library {
  id: string;
  name: string;
  category: Category;
  icon: string;
  shortDescription: string;
  officialUrl?: string;
}

export interface LibraryDetails {
  name: string;
  description: string;
  useCases: string[];
  codeExample: string;
  keyFeatures: string[];
}
