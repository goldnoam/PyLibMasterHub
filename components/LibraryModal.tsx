
import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { Library } from '../types';

declare global {
  interface Window {
    Prism: any;
  }
}

interface LibraryModalProps {
  library: Library | null;
  onClose: () => void;
}

const LibraryModal: React.FC<LibraryModalProps> = ({ library, onClose }) => {
  const [copied, setCopied] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  useEffect(() => {
    if (library && window.Prism) {
      const timer = setTimeout(() => {
        try {
          window.Prism.highlightAll();
        } catch (e) {
          console.warn("Prism highlighting failed:", e);
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [library]);

  const handleCopy = () => {
    if (library?.codeExample) {
      navigator.clipboard.writeText(library.codeExample);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleSpeak = useCallback(() => {
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    if (!library) return;
    const textToSpeak = `${library.name}. ${library.longDescription || library.shortDescription}. This is a ${library.category} library.`;
    
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    
    setIsSpeaking(true);
    window.speechSynthesis.speak(utterance);
  }, [library, isSpeaking]);

  const lineNumbers = useMemo(() => {
    if (!library?.codeExample) return [];
    return library.codeExample.split('\n').map((_, i) => i + 1);
  }, [library?.codeExample]);

  if (!library) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 sm:p-6 lg:p-8">
      <div 
        className="absolute inset-0 bg-slate-950/70 backdrop-blur-md transition-opacity cursor-pointer" 
        onClick={onClose}
      />
      
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-white dark:bg-slate-900 rounded-3xl shadow-2xl overflow-hidden flex flex-col border border-slate-200 dark:border-slate-800 animate-in zoom-in-95 duration-200">
        {/* Modal Header */}
        <div className="px-6 py-5 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center sticky top-0 bg-white dark:bg-slate-900 z-10">
          <div className="flex items-center space-x-4">
            <span className="text-4xl">{library.icon}</span>
            <div>
              <div className="flex items-center space-x-3">
                <h2 className="text-2xl font-bold text-slate-900 dark:text-white leading-tight">
                  {library.name}
                </h2>
                <span className="flex items-center space-x-1 px-2 py-0.5 bg-brand-100 dark:bg-brand-900/30 text-brand-700 dark:text-brand-400 text-[10px] font-black uppercase rounded-md border border-brand-200 dark:border-brand-800">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                  <span>Offline Ready</span>
                </span>
              </div>
              <span className="text-[10px] font-black uppercase tracking-widest text-brand-600 dark:text-brand-400">
                {library.category}
              </span>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button 
              onClick={handleSpeak}
              className={`p-3 rounded-xl transition-all flex items-center space-x-2 font-bold text-xs uppercase tracking-widest ${
                isSpeaking 
                ? 'bg-brand-500 text-white animate-pulse' 
                : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}
              title={isSpeaking ? "Stop Speaking" : "Read Aloud"}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {isSpeaking ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"></path>
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"></path>
                )}
              </svg>
              <span className="hidden sm:inline">{isSpeaking ? 'Stop' : 'Speak'}</span>
            </button>
            
            <button 
              onClick={onClose}
              className="p-3 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors text-slate-500"
              aria-label="Close modal"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="flex-grow overflow-y-auto p-6 sm:p-8 scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-slate-700">
          <div className="space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Overview */}
            <section>
              <h3 className="text-lg font-bold mb-4 flex items-center dark:text-white">
                <span className="w-2 h-6 bg-brand-500 rounded-full mr-3"></span>
                Overview
              </h3>
              <p className="text-slate-600 dark:text-slate-400 leading-relaxed text-lg">
                {library.longDescription || library.shortDescription}
              </p>
            </section>

            {/* Key Features */}
            {library.keyFeatures && (
              <section className="bg-slate-50 dark:bg-slate-800/50 p-6 rounded-3xl border border-slate-100 dark:border-slate-800">
                <h3 className="text-sm font-black mb-5 uppercase tracking-widest text-slate-900 dark:text-white flex items-center">
                  <svg className="w-4 h-4 mr-2 text-brand-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                  Key Capabilities
                </h3>
                <ul className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {library.keyFeatures.map((f, i) => (
                    <li key={i} className="flex items-start">
                      <span className="w-5 h-5 bg-green-500/10 text-green-600 rounded flex items-center justify-center mr-3 text-[10px] flex-shrink-0">✓</span>
                      <span className="text-slate-600 dark:text-slate-400 text-sm">{f}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {/* Code Snippet */}
            {library.codeExample && (
              <section>
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
                  <h3 className="text-lg font-bold dark:text-white">Implementation Guide</h3>
                  <button 
                    onClick={handleCopy}
                    className={`group flex items-center space-x-2 px-6 py-2.5 rounded-2xl font-bold text-sm transition-all shadow-lg active:scale-95 ${
                      copied 
                      ? 'bg-green-600 text-white shadow-green-500/20' 
                      : 'bg-brand-600 hover:bg-brand-700 text-white shadow-brand-500/30'
                    }`}
                  >
                    <svg className={`w-4 h-4 transition-transform ${copied ? 'scale-125' : 'group-hover:rotate-12'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      {copied ? (
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                      ) : (
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"></path>
                      )}
                    </svg>
                    <span>{copied ? 'Copied Successfully' : 'Copy Source Code'}</span>
                  </button>
                </div>
                
                <div className="rounded-3xl overflow-hidden shadow-2xl bg-[#282a36] border border-slate-800 group/code">
                  <div className="flex">
                    {/* Line Number Gutter */}
                    <div className="hidden sm:flex flex-col bg-[#21222c] text-slate-500/50 text-right pr-4 pl-4 py-6 select-none font-mono text-sm border-r border-slate-800/40">
                      {lineNumbers.map((num) => (
                        <span key={num} className="leading-relaxed h-[1.5rem]">{num}</span>
                      ))}
                    </div>
                    {/* Code Content */}
                    <pre className="flex-grow !m-0 !p-6 overflow-x-auto text-sm leading-relaxed scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                      <code className="language-python !bg-transparent">
                        {library.codeExample}
                      </code>
                    </pre>
                  </div>
                </div>
              </section>
            )}

            {/* Official Documentation Link */}
            {library.officialUrl && (
              <div className="text-center pt-8 border-t border-slate-100 dark:border-slate-800">
                <a 
                  href={library.officialUrl} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center px-8 py-3 bg-brand-600 text-white rounded-2xl font-bold hover:bg-brand-700 transition-all shadow-xl shadow-brand-500/20 hover:-translate-y-0.5"
                >
                  Visit {library.name} Project Site
                  <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2-2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
                </a>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LibraryModal;
