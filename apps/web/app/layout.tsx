import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "IPC Classifier",
  description: "Abstract-to-IPC candidate classification with evidence and calibrated confidence.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col">
        <nav className="border-b border-zinc-200 px-6 py-3 text-sm dark:border-zinc-800">
          <Link href="/" className="mr-4 font-semibold">IPC Classifier</Link>
          <Link href="/taxonomy" className="mr-4 underline">Taxonomy</Link>
          <Link href="/status" className="underline">Source status</Link>
        </nav>
        {children}
      </body>
    </html>
  );
}
