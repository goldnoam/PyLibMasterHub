
import React from 'react';
import { Library } from '../types';

interface LibraryCardProps {
  library: Library;
  onClick: (lib: Library) => void;
}

const LibraryCard: React.FC<LibraryCardProps> = ({ library, onClick }) => {
  return (
    <button
      onClick={() => onClick(library)}
      className="group relative flex flex-col p-6 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 hover:border-brand-500 dark:hover:border-brand-500/50 hover:shadow-xl hover:shadow-brand-500/10 transition-all duration-300 text-left overflow-hidden"
    >
      <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
        <span className="text-4xl">{library.icon}</span>
      </div>
      
      <div className="flex items-center space-x-3 mb-4">
        <span className="text-2xl">{library.icon}</span>
        <h3 className="text-lg font-bold text-slate-900 dark:text-white group-hover:text-brand-500 transition-colors">
          {library.name}
        </h3>
      </div>
      
      <p className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2">
        {library.shortDescription}
      </p>
      
      <div className="mt-6 flex items-center text-xs font-semibold text-brand-600 dark:text-brand-400 uppercase tracking-wider">
        Learn More
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>
    </button>
  );
};

export default LibraryCard;
