import React, { useEffect, useState, useCallback } from 'react';
import { Library, LibraryDetails } from '../types';
import { fetchLibraryDetails } from '../services/gemini';

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
  const [details, setDetails] = useState<LibraryDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isOffline, setIsOffline] = useState(!navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  useEffect(() => {
    if (library) {
      if (isOffline) {
        setLoading(false);
        setDetails(null);
        return;
      }

      setLoading(true);
      setError(null);
      setCopied(false);
      setDetails(null); 
      
      fetchLibraryDetails(library.name)
        .then((data) => {
          setDetails(data);
        })
        .catch((err) => {
            console.error("Modal Data Fetch Error:", err);
            setError("Unable to connect to AI documentation service.");
        })
        .finally(() => setLoading(false));
    } else {
      setDetails(null);
      setError(null);
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  }, [library, isOffline]);

  useEffect(() => {
    if (details && window.Prism) {
      const timer = setTimeout(() => {
        try {
          window.Prism.highlightAll();
        } catch (e) {
          console.warn("Prism highlighting failed:", e);
        }
      }, 200);
      return () => clearTimeout(timer);
    }
  }, [details]);

  const handleCopy = () => {
    if (details?.codeExample) {
      navigator.clipboard.writeText(details.codeExample);
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

    const textToSpeak = details 
      ? `${details.name}. ${details.description}. Key features include: ${details.keyFeatures.join(', ')}.`
      : library ? `${library.name}. ${library.shortDescription}. This is a ${library.category} library.` : "";
    
    if (!textToSpeak) return;

    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    
    setIsSpeaking(true);
    window.speechSynthesis.speak(utterance);
  }, [details, library, isSpeaking]);

  if (!library) return null;

  const renderSafeString = (val: any): string => {
    if (typeof val === 'string') return val;
    if (val === null || val === undefined) return '';
    return String(val);
  };

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
            <span className="text-4xl">{renderSafeString(library.icon)}</span>
            <div>
              <div className="flex items-center space-x-3">
                <h2 className="text-2xl font-bold text-slate-900 dark:text-white leading-tight">
                  {renderSafeString(library.name)}
                </h2>
                {isOffline && (
                  <span className="flex items-center space-x-1 px-2 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-[10px] font-black uppercase rounded-md border border-amber-200 dark:border-amber-800">
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192l-3.536 3.536M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-5 0a4 4 0 11-8 0 4 4 0 018 0z"></path></svg>
                    <span>Offline View</span>
                  </span>
                )}
              </div>
              <span className="text-[10px] font-black uppercase tracking-widest text-brand-600 dark:text-brand-400">
                {renderSafeString(library.category)}
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

        {/* Modal Content */}
        <div className="flex-grow overflow-y-auto p-6 sm:p-8 scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-slate-700">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-24 space-y-4">
              <div className="w-12 h-12 border-4 border-brand-500 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-slate-500 font-medium">Fetching details from AI...</p>
            </div>
          ) : (
            <div className="space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {/* Notification Banner for Connection Issues */}
              {(isOffline || error) && (
                <div className="flex items-start space-x-4 p-4 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/50 rounded-2xl">
                  <div className="text-amber-600 dark:text-amber-400 pt-1">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-amber-900 dark:text-amber-200">Limited Documentation View</h4>
                    <p className="text-xs text-amber-700 dark:text-amber-400 mt-1 leading-relaxed">
                      {isOffline 
                        ? "Full AI-powered deep dives, code examples, and technical specifications are available when online. Showing local summary instead."
                        : error}
                    </p>
                  </div>
                </div>
              )}

              {/* Overview Section */}
              <section>
                <h3 className="text-lg font-bold mb-4 flex items-center dark:text-white">
                  <span className="w-2 h-6 bg-brand-500 rounded-full mr-3"></span>
                  Overview
                </h3>
                <p className="text-slate-600 dark:text-slate-400 leading-relaxed text-lg italic sm:not-italic">
                  {details ? renderSafeString(details.description) : renderSafeString(library.shortDescription)}
                </p>
              </section>

              {/* Structured Data Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <section className="bg-slate-50 dark:bg-slate-800/50 p-6 rounded-3xl border border-slate-100 dark:border-slate-800">
                  <h3 className="text-sm font-black mb-5 uppercase tracking-widest text-slate-900 dark:text-white flex items-center">
                    <svg className="w-4 h-4 mr-2 text-brand-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    Key Features
                  </h3>
                  <ul className="space-y-4">
                    {details && Array.isArray(details.keyFeatures) ? (
                      details.keyFeatures.map((f, i) => (
                        <li key={i} className="flex items-start">
                          <span className="w-5 h-5 bg-green-500/10 text-green-600 rounded flex items-center justify-center mr-3 text-[10px] flex-shrink-0">✓</span>
                          <span className="text-slate-600 dark:text-slate-400 text-sm">{renderSafeString(f)}</span>
                        </li>
                      ))
                    ) : (
                      <li className="flex items-start opacity-50 italic">
                         <span className="text-slate-500 text-xs">Features available in Online Mode...</span>
                      </li>
                    )}
                  </ul>
                </section>
                
                <section className="bg-slate-50 dark:bg-slate-800/50 p-6 rounded-3xl border border-slate-100 dark:border-slate-800">
                  <h3 className="text-sm font-black mb-5 uppercase tracking-widest text-slate-900 dark:text-white flex items-center">
                    <svg className="w-4 h-4 mr-2 text-brand-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                    Common Use Cases
                  </h3>
                  <ul className="space-y-4">
                    {details && Array.isArray(details.useCases) ? (
                      details.useCases.map((u, i) => (
                        <li key={i} className="flex items-start">
                          <span className="w-5 h-5 bg-brand-500/10 text-brand-600 rounded flex items-center justify-center mr-3 text-[10px] flex-shrink-0">●</span>
                          <span className="text-slate-600 dark:text-slate-400 text-sm">{renderSafeString(u)}</span>
                        </li>
                      ))
                    ) : (
                      <li className="flex items-start opacity-50 italic">
                         <span className="text-slate-500 text-xs">Use cases available in Online Mode...</span>
                      </li>
                    )}
                  </ul>
                </section>
              </div>

              {/* Code Example Section (Conditional or Placeholder) */}
              <section>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-bold dark:text-white">Code Snippet</h3>
                  {details && (
                    <button 
                      onClick={handleCopy}
                      className={`flex items-center space-x-2 text-[10px] font-bold uppercase tracking-widest px-4 py-2 rounded-xl transition-all border shadow-sm ${
                        copied 
                        ? 'bg-green-500 border-green-500 text-white' 
                        : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:border-brand-500'
                      }`}
                    >
                      {copied ? 'Copied!' : 'Copy Code'}
                    </button>
                  )}
                </div>
                
                {details ? (
                  <div className="rounded-2xl overflow-hidden shadow-2xl bg-[#282a36] border border-slate-800">
                    <pre className="!m-0 !p-6 overflow-x-auto text-sm leading-relaxed scrollbar-thin scrollbar-thumb-slate-700">
                      <code className="language-python">
                        {renderSafeString(details.codeExample)}
                      </code>
                    </pre>
                  </div>
                ) : (
                  <div className="h-32 bg-slate-100 dark:bg-slate-800/30 rounded-2xl flex flex-col items-center justify-center border border-dashed border-slate-300 dark:border-slate-800 text-slate-400 text-center px-4">
                    <svg className="w-8 h-8 mb-2 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"></path></svg>
                    <p className="text-xs font-medium">Technical implementation examples require an active connection.</p>
                  </div>
                )}
              </section>

              {/* Official Documentation Link */}
              {library.officialUrl && (
                <div className="text-center pt-8 border-t border-slate-100 dark:border-slate-800">
                  <a 
                    href={library.officialUrl} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-8 py-3 bg-brand-600 text-white rounded-2xl font-bold hover:bg-brand-700 transition-all shadow-xl shadow-brand-500/20 hover:-translate-y-0.5"
                  >
                    Go to {library.name} Documentation
                    <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
                  </a>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LibraryModal;