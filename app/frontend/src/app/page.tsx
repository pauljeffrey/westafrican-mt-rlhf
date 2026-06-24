"use client";

import { useState, useRef } from "react";
import { LANGUAGES, EXAMPLES, type Language, type Example } from "@/lib/data";
import { translateText, ApiError } from "@/lib/api";

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

function IconGlobe() {
  return (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth={1.6} viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="10" />
      <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

function IconCopy() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
      <rect width="13" height="13" x="9" y="9" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function IconCheck() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IconArrowRight() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path d="M5 12h14M12 5l7 7-7 7" />
    </svg>
  );
}

function IconClear() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function LanguageTab({
  lang,
  active,
  onClick,
}: {
  lang: Language;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium
        transition-all duration-150 whitespace-nowrap border
        ${
          active
            ? "bg-forest-800 text-white border-forest-800 shadow-sm"
            : "bg-white text-stone-600 border-stone-200 hover:border-forest-700 hover:text-forest-800"
        }
      `}
    >
      <span>{lang.flag}</span>
      <span>{lang.name}</span>
    </button>
  );
}

function ExampleChip({ example, onClick }: { example: Example; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="group flex items-start gap-2 w-full text-left px-4 py-3 rounded-xl
        border border-stone-200 bg-white hover:border-amber-400 hover:bg-amber-50
        transition-all duration-150"
    >
      <span className="mt-0.5 flex-shrink-0 w-5 h-5 rounded-full bg-amber-100 group-hover:bg-amber-200
        text-amber-700 flex items-center justify-center transition-colors duration-150">
        <IconArrowRight />
      </span>
      <span className="flex-1 min-w-0">
        <span className="block text-xs font-medium text-amber-600 mb-0.5">{example.label}</span>
        <span className="block text-sm text-stone-700 leading-snug">{example.en}</span>
      </span>
    </button>
  );
}

function TranslationSkeleton() {
  return (
    <div className="space-y-2.5 py-1">
      <div className="h-4 rounded shimmer w-full" />
      <div className="h-4 rounded shimmer w-4/5" />
      <div className="h-4 rounded shimmer w-2/3" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

const MAX_CHARS = 500;

export default function Home() {
  const [activeLang, setActiveLang] = useState<Language>(LANGUAGES[0]);
  const [sourceText, setSourceText] = useState("");
  const [translation, setTranslation] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const selectedLang = LANGUAGES.find((l) => l.code === activeLang.code) ?? LANGUAGES[0];
  const examples = EXAMPLES[selectedLang.code] ?? [];

  const handleTranslate = async () => {
    const text = sourceText.trim();
    if (!text || loading) return;
    setLoading(true);
    setError("");
    setTranslation("");
    try {
      const res = await translateText(text, selectedLang.code);
      setTranslation(res.translation);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.status === 0 ? "API URL not configured." : `Error ${err.status}: ${err.message}`);
      } else {
        setError("Translation failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleExample = (example: Example) => {
    setSourceText(example.en);
    setTranslation("");
    setError("");
    textareaRef.current?.focus();
  };

  const handleCopy = async () => {
    if (!translation) return;
    await navigator.clipboard.writeText(translation);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleClear = () => {
    setSourceText("");
    setTranslation("");
    setError("");
    textareaRef.current?.focus();
  };

  const charCount = sourceText.length;
  const overLimit = charCount > MAX_CHARS;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-forest-800 text-white px-6 py-5 shadow-md">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-green-200">
              <IconGlobe />
            </span>
            <div>
              <h1 className="text-xl font-bold leading-tight tracking-tight">
                West African Translator
              </h1>
              <p className="text-green-200 text-xs mt-0.5">
                Powered by <span className="font-semibold">BeardedMonster/gemma-270m-translate-it</span>
              </p>
            </div>
          </div>
          <a
            href="https://huggingface.co/BeardedMonster/gemma-270m-translate-it"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex items-center gap-2 text-xs text-green-200 hover:text-white
              border border-green-600 hover:border-white rounded-lg px-3 py-1.5 transition-colors"
          >
            🤗 View on HuggingFace
          </a>
        </div>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
        {/* Language selector */}
        <section>
          <p className="text-xs font-semibold text-stone-400 uppercase tracking-widest mb-3">
            Translate English to
          </p>
          <div className="flex flex-wrap gap-2">
            {LANGUAGES.map((lang) => (
              <LanguageTab
                key={lang.code}
                lang={lang}
                active={lang.code === selectedLang.code}
                onClick={() => {
                  setActiveLang(lang);
                  setTranslation("");
                  setError("");
                }}
              />
            ))}
          </div>
          <p className="mt-1.5 text-xs text-stone-400">{selectedLang.region}</p>
        </section>

        {/* Translation panel */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Source */}
          <div className="flex flex-col rounded-2xl border border-stone-200 bg-white overflow-hidden shadow-sm">
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-stone-100 bg-stone-50">
              <span className="text-xs font-semibold text-stone-500 uppercase tracking-wider">English</span>
              {sourceText && (
                <button
                  onClick={handleClear}
                  className="text-stone-400 hover:text-stone-700 transition-colors"
                  title="Clear"
                >
                  <IconClear />
                </button>
              )}
            </div>
            <textarea
              ref={textareaRef}
              value={sourceText}
              onChange={(e) => {
                setSourceText(e.target.value);
                if (translation) setTranslation("");
                if (error) setError("");
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleTranslate();
              }}
              placeholder="Enter text to translate… (Ctrl+Enter to translate)"
              rows={6}
              maxLength={MAX_CHARS + 50}
              className="flex-1 resize-none px-4 py-3 text-sm text-stone-800 placeholder-stone-300
                focus:outline-none leading-relaxed"
            />
            <div className="flex items-center justify-between px-4 py-2.5 border-t border-stone-100 bg-stone-50">
              <span
                className={`text-xs tabular-nums ${overLimit ? "text-red-500 font-semibold" : "text-stone-400"}`}
              >
                {charCount} / {MAX_CHARS}
              </span>
              <button
                onClick={handleTranslate}
                disabled={!sourceText.trim() || loading || overLimit}
                className="flex items-center gap-2 bg-forest-800 hover:bg-forest-700
                  disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium
                  px-4 py-1.5 rounded-lg transition-colors duration-150"
              >
                {loading ? (
                  <>
                    <span className="w-3.5 h-3.5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                    Translating…
                  </>
                ) : (
                  <>
                    Translate
                    <IconArrowRight />
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Target */}
          <div className="flex flex-col rounded-2xl border border-stone-200 bg-white overflow-hidden shadow-sm">
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-stone-100 bg-stone-50">
              <span className="text-xs font-semibold text-stone-500 uppercase tracking-wider">
                {selectedLang.flag} {selectedLang.name}{" "}
                <span className="font-normal text-stone-400">({selectedLang.nativeName})</span>
              </span>
              {translation && (
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1 text-xs text-stone-400 hover:text-stone-700 transition-colors"
                >
                  {copied ? (
                    <><IconCheck /><span>Copied</span></>
                  ) : (
                    <><IconCopy /><span>Copy</span></>
                  )}
                </button>
              )}
            </div>
            <div className="flex-1 px-4 py-3 min-h-[144px]">
              {loading ? (
                <TranslationSkeleton />
              ) : error ? (
                <p className="text-sm text-red-500 leading-relaxed animate-fade-in">{error}</p>
              ) : translation ? (
                <p className="text-sm text-stone-800 leading-relaxed animate-fade-in whitespace-pre-wrap">
                  {translation}
                </p>
              ) : (
                <p className="text-sm text-stone-300 select-none">Translation will appear here…</p>
              )}
            </div>
            <div className="px-4 py-2.5 border-t border-stone-100 bg-stone-50 min-h-[40px]" />
          </div>
        </section>

        {/* Examples */}
        <section>
          <p className="text-xs font-semibold text-stone-400 uppercase tracking-widest mb-3">
            Try an example — {selectedLang.name}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {examples.map((ex, i) => (
              <ExampleChip key={i} example={ex} onClick={() => handleExample(ex)} />
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-stone-200 bg-white px-6 py-4">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-stone-400">
          <span>
            Model fine-tuned on{" "}
            <a
              href="https://huggingface.co/datasets/Aletheia-ng/tds-sft"
              className="text-forest-700 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Aletheia-ng/tds-sft
            </a>{" "}
            with AfriCOMET-guided RLHF
          </span>
          <span className="flex items-center gap-3">
            <a
              href="https://github.com/BeardedMonster/west-african-mt-rlhf"
              className="hover:text-stone-700 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
            <a
              href="https://huggingface.co/BeardedMonster/gemma-270m-translate-it"
              className="hover:text-stone-700 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              HuggingFace
            </a>
          </span>
        </div>
      </footer>
    </div>
  );
}
