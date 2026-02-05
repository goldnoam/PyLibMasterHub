import React, { useState, useMemo, useEffect } from 'react';
import Layout, { FontSize } from './components/Layout';
import LibraryCard from './components/LibraryCard';
import LibraryModal from './components/LibraryModal';
import { LIBRARIES, CATEGORIES } from './constants';
import { Library, Category } from './types';

const App: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<Category | 'All'>('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLibrary, setSelectedLibrary] = useState<Library | null>(null);
  const [isDark, setIsDark] = useState(true);
  
  // Initialize font size from localStorage if available
  const [fontSize, setFontSize] = useState<FontSize>(() => {
    const saved = localStorage.getItem('pylib_font_size');
    return (saved === 'sm' || saved === 'base' || saved === 'lg') ? saved : 'base';
  });

  // Sync dark mode class and handle dynamic metadata/favicon injection
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }

    // SEO and Meta Tags Injection
    document.title = "PyLibMaster Hub | The Ultimate Python Library Guide";
    
    // Favicon Injection
    const link = document.querySelector("link[rel~='icon']") || document.createElement('link');
    // @ts-ignore
    link.rel = 'icon';
    // @ts-ignore
    link.href = 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🐍</text></svg>';
    if (!document.querySelector("link[rel~='icon']")) {
      document.head.appendChild(link);
    }
  }, [isDark]);

  // Persist font size choice
  useEffect(() => {
    localStorage.setItem('pylib_font_size', fontSize);
  }, [fontSize]);

  const filteredLibraries = useMemo(() => {
    return LIBRARIES.filter(lib => {
      const matchesCategory = selectedCategory === 'All' || lib.category === selectedCategory;
      const matchesSearch = lib.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          lib.shortDescription.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesCategory && matchesSearch;
    });
  }, [selectedCategory, searchQuery]);

  return (
    <Layout 
      isDark={isDark} 
      toggleTheme={() => setIsDark(!isDark)}
      fontSize={fontSize}
      setFontSize={setFontSize}
    >
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-12 pb-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center relative z-10">
          <div className="inline-flex items-center space-x-2 bg-brand-500/10 dark:bg-brand-500/20 px-3 py-1 rounded-full text-brand-600 dark:text-brand-400 text-sm font-medium mb-6">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-500"></span>
            </span>
            <span>Comprehensive Python Ecosystem Index</span>
          </div>
          
          <h1 className="text-5xl sm:text-7xl font-extrabold text-slate-900 dark:text-white mb-6 tracking-tight leading-tight">
            Master Every Python <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-500 to-indigo-500">
              Package That Matters
            </span>
          </h1>
          
          <p className="max-w-2xl mx-auto text-xl text-slate-600 dark:text-slate-400 mb-10 leading-relaxed">
            Explore Data Science, Web Scraping, Machine Learning, and foundational libraries with interactive examples and deep technical insights.
          </p>

          <div className="max-w-xl mx-auto relative group">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <svg className="h-5 w-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
            </div>
            <input 
              type="text" 
              placeholder="Search library, keyword or category..."
              className="w-full pl-12 pr-4 py-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 focus:outline-none focus:ring-2 focus:ring-brand-500/50 shadow-lg transition-all"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>
        
        {/* Background Gradients */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full -z-0 pointer-events-none opacity-20 dark:opacity-40">
           <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-brand-500/30 rounded-full blur-[120px]"></div>
           <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-indigo-500/20 rounded-full blur-[120px]"></div>
        </div>
      </section>

      {/* Main Catalog */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Filter Bar */}
        <div className="flex flex-wrap items-center gap-3 mb-12 sticky top-20 z-40 bg-slate-50/80 dark:bg-slate-950/80 backdrop-blur-md py-4 rounded-xl px-2">
          <button
            onClick={() => setSelectedCategory('All')}
            className={`px-4 py-2 rounded-full text-sm font-semibold transition-all ${
              selectedCategory === 'All' 
              ? 'bg-brand-600 text-white shadow-lg shadow-brand-500/20' 
              : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800'
            }`}
          >
            All
          </button>
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-4 py-2 rounded-full text-sm font-semibold transition-all whitespace-nowrap ${
                selectedCategory === cat 
                ? 'bg-brand-600 text-white shadow-lg shadow-brand-500/20' 
                : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Results Info */}
        <div className="mb-8 flex justify-between items-center">
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">
            {selectedCategory === 'All' ? 'Full Catalog' : selectedCategory} 
            <span className="ml-2 text-slate-400 font-normal">({filteredLibraries.length} libraries found)</span>
          </h2>
        </div>

        {/* Grid */}
        {filteredLibraries.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredLibraries.map(lib => (
              <LibraryCard 
                key={lib.id} 
                library={lib} 
                onClick={setSelectedLibrary} 
              />
            ))}
          </div>
        ) : (
          <div className="py-32 text-center">
            <div className="inline-block p-6 bg-slate-100 dark:bg-slate-900 rounded-full mb-6 text-4xl">🔎</div>
            <h3 className="text-xl font-bold mb-2">No libraries found matching your criteria.</h3>
            <p className="text-slate-500">Try searching for something else or changing the category.</p>
          </div>
        )}

        {/* Ad Placeholder */}
        <div className="my-16 w-full flex flex-col items-center">
          <p className="text-slate-400 text-[10px] font-mono tracking-widest uppercase mb-2">Resource Placeholder</p>
          <div className="w-full h-32 bg-slate-100 dark:bg-slate-900 rounded-3xl flex items-center justify-center border border-dashed border-slate-300 dark:border-slate-800 overflow-hidden text-slate-400 text-sm italic">
             Community-driven content coming soon
          </div>
        </div>
      </section>

      {/* Community Callout */}
      <section className="mt-32 mb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto rounded-[3rem] p-12 sm:p-20 bg-gradient-to-br from-brand-600 to-indigo-700 relative overflow-hidden text-center text-white shadow-2xl">
          <div className="relative z-10">
            <h2 className="text-4xl sm:text-5xl font-bold mb-6">Contribute to the Index</h2>
            <p className="text-xl text-brand-100 mb-10 max-w-2xl mx-auto leading-relaxed">
              Missing a library? Our index is growing and we value community input.
            </p>
            <a 
              href="mailto:goldnoamai@gmail.com"
              className="inline-flex items-center px-8 py-4 bg-white text-brand-700 font-bold rounded-2xl hover:bg-brand-50 transition-all shadow-lg hover:-translate-y-1"
            >
              Contact Support
              <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
            </a>
          </div>
          {/* Decorative shapes */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2 blur-3xl"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-black/10 rounded-full translate-y-1/2 -translate-x-1/2 blur-3xl"></div>
        </div>
      </section>

      {/* Modals */}
      <LibraryModal 
        library={selectedLibrary} 
        onClose={() => setSelectedLibrary(null)} 
      />
    </Layout>
  );
};

export default App;