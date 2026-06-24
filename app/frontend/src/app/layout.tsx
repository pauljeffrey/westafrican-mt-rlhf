import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "West African Translator",
  description:
    "Translate English into West African languages — Hausa, Igbo, Yoruba, Wolof, Ewe, Fon, and Twi — powered by a fine-tuned Gemma model.",
  openGraph: {
    title: "West African Translator",
    description: "English → Hausa · Igbo · Yoruba · Wolof · Ewe · Fon · Twi",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="bg-stone-50 text-stone-900 min-h-screen">{children}</body>
    </html>
  );
}
