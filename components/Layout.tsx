import React, { useState, useEffect } from 'react';

export type FontSize = 'sm' | 'base' | 'lg';

interface LayoutProps {
  children: React.ReactNode;
  toggleTheme: () => void;
  isDark: boolean;
  fontSize: FontSize;
  setFontSize: (size: FontSize) => void;
}

const Layout: React.FC<LayoutProps> = ({ children, toggleTheme, isDark, fontSize, setFontSize }) => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const fontSizeClasses = {
    sm: 'text-sm',
    base: 'text-base',
    lg: 'text-lg',
  };

  const getAriaLabel = (size: FontSize) => {
    switch(size) {
      case 'sm': return 'Switch to small text';
      case 'base': return 'Switch to medium text';
      case 'lg': return 'Switch to large text';
      default: return '';
    }
  };

  return (
    <div className={`min-h-screen flex flex-col font-sans selection:bg-brand-500 selection:text-white transition-all duration-200 ${fontSizeClasses[fontSize]}`}>
      {/* Header */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 border-b ${
        scrolled 
          ? 'bg-white/80 dark:bg-slate-950/80 backdrop-blur-md py-3 border-slate-200 dark:border-slate-800' 
          : 'bg-transparent py-5 border-transparent'
      }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-brand-600 rounded-lg flex items-center justify-center text-white font-bold shadow-lg shadow-brand-500/20">
              Py
            </div>
            <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-white hidden sm:block">
              LibMaster <span className="text-brand-500">Hub</span>
            </span>
          </div>
          
          <div className="flex items-center space-x-2 sm:space-x-4">
            {/* Font Size Controls */}
            <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1" role="group" aria-label="Adjust font size">
              {(['sm', 'base', 'lg'] as FontSize[]).map((size) => (
                <button
                  key={size}
                  onClick={() => setFontSize(size)}
                  aria-label={getAriaLabel(size)}
                  aria-pressed={fontSize === size}
                  className={`px-3 py-1 rounded-md text-xs font-bold uppercase transition-all ${
                    fontSize === size 
                      ? 'bg-white dark:bg-slate-700 text-brand-600 dark:text-brand-400 shadow-sm' 
                      : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
                >
                  {size === 'base' ? 'M' : size.toUpperCase()}
                </button>
              ))}
            </div>

            <button
              onClick={toggleTheme}
              className="p-2 rounded-full hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors"
              title="Toggle Theme"
              aria-label="Toggle dark and light mode"
            >
              {isDark ? (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 9h-1m15.364-6.364l-.707.707M6.343 17.657l-.707.707m12.728 0l-.707-.707M6.343 6.343l-.707-.707M12 5a7 7 0 100 14 7 7 0 000-14z" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-grow pt-24 pb-16">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-6 md:space-y-0">
            <div className="text-center md:text-left">
               <p className="text-lg font-bold text-slate-900 dark:text-white mb-2">(C) Noam Gold AI 2026</p>
               <p className="text-slate-500 dark:text-slate-400">The premier destination for Python library documentation and exploration.</p>
            </div>
            
            <div className="flex flex-col items-center md:items-end space-y-4">
              <a 
                href="mailto:goldnoamai@gmail.com" 
                className="group flex items-center space-x-3 bg-brand-500/10 hover:bg-brand-500 text-brand-600 dark:text-brand-400 hover:text-white px-6 py-3 rounded-2xl transition-all font-bold shadow-sm"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
                </svg>
                <span>Send Feedback "goldnoamai@gmail.com"</span>
              </a>
              <div className="flex space-x-6 text-sm text-slate-500">
                <a href="#" className="hover:text-brand-500 transition-colors">Privacy</a>
                <a href="#" className="hover:text-brand-500 transition-colors">Terms</a>
                <a href="#" className="hover:text-brand-500 transition-colors">Sitemap</a>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;