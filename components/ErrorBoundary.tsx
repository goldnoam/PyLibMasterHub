
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

/**
 * ErrorBoundary catches JavaScript errors in child components.
 * It follows the official React documentation for class-based error boundaries.
 */
// Fix: Use 'Component' from 'react' explicitly to ensure the base class properties like 'props' and 'state' are correctly typed and recognized by the TypeScript compiler.
class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  // The lifecycle method getDerivedStateFromError is used to update state after an error is thrown.
  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log the error for debugging purposes
    console.error("Uncaught error caught by ErrorBoundary:", error, errorInfo);
  }

  public render(): ReactNode {
    // Fix: Destructure state and props from 'this' inside render to cleanly access hasError, error, and children.
    const { hasError, error } = this.state;
    const { children } = this.props;

    if (hasError) {
      // Fallback UI when an error occurs
      return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950 p-6">
          <div className="max-w-md w-full bg-white dark:bg-slate-900 rounded-3xl p-8 shadow-2xl border border-red-100 dark:border-red-900/30 text-center">
            <div className="w-16 h-16 bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-full flex items-center justify-center mx-auto mb-6 text-2xl font-bold">
              !
            </div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">Something went wrong</h1>
            <p className="text-slate-600 dark:text-slate-400 mb-8">
              We encountered an unexpected error while rendering the interface.
            </p>
            <button
              className="w-full py-3 bg-brand-600 hover:bg-brand-700 text-white font-bold rounded-2xl transition-colors shadow-lg shadow-brand-500/20"
              onClick={() => window.location.reload()}
            >
              Reload Application
            </button>
            {process.env.NODE_ENV !== 'production' && error && (
              <pre className="mt-8 p-4 bg-slate-100 dark:bg-slate-800 rounded-xl text-left text-xs overflow-auto max-h-40 text-red-500">
                {error.toString()}
              </pre>
            )}
          </div>
        </div>
      );
    }

    // Accessing 'children' through the destructured props variable.
    return children;
  }
}

export default ErrorBoundary;
