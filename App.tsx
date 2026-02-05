
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
  
  // Initialize theme from localStorage
  const [isDark, setIsDark] = useState<boolean>(() => {
    const saved = localStorage.getItem('pylib_theme');
    return saved !== null ? saved === 'true' : true;
  });
  
  // Initialize font size from localStorage
  const [fontSize, setFontSize] = useState<FontSize>(() => {
    const saved = localStorage.getItem('pylib_font_size');
    return (saved === 'sm' || saved === 'base' || saved === 'lg') ? saved : 'base';
  });

  // Sync dark mode class and theme choice
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('pylib_theme', isDark.toString());
  }, [isDark]);

  // Handle dynamic metadata/favicon injection
  useEffect(() => {
    document.title = "PyLibMaster Hub | The Ultimate Python Library Guide";
    const link = document.querySelector("link[rel~='icon']") || document.createElement('link');
    // @ts-ignore
    link.rel = 'icon';
    // @ts-ignore
    link.href = 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🐍</text></svg>';
    if (!document.querySelector("link[rel~='icon']")) {
      document.head.appendChild(link);
    }
  }, []);

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
            Learn Python Libraries<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-500 to-indigo-500">
              Offline Mastery Hub
            </span>
          </h1>
          
          <p className="max-w-2xl mx-auto text-xl text-slate-600 dark:text-slate-400 mb-10 leading-relaxed">
            Instant access to Data Science, ML, and automation libraries with static examples and high-quality documentation.
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
        
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full -z-0 pointer-events-none opacity-20 dark:opacity-40">
           <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-brand-500/30 rounded-full blur-[120px]"></div>
           <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-indigo-500/20 rounded-full blur-[120px]"></div>
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
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

        <div className="mb-8 flex justify-between items-center">
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">
            {selectedCategory === 'All' ? 'Full Catalog' : selectedCategory} 
            <span className="ml-2 text-slate-400 font-normal">({filteredLibraries.length} libraries found)</span>
          </h2>
        </div>

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
      </section>

      <LibraryModal 
        library={selectedLibrary} 
        onClose={() => setSelectedLibrary(null)} 
      />
    </Layout>
  );
};

export default App;
