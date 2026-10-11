import { Suspense } from "react";
import { TaxonomyBrowser } from "./TaxonomyBrowser";

export default function TaxonomyPage() {
  return (
    <main className="mx-auto w-full max-w-5xl space-y-4 p-6">
      <h1 className="text-2xl font-semibold">Taxonomy browser</h1>
      <Suspense fallback={<p>Loading…</p>}>
        <TaxonomyBrowser />
      </Suspense>
    </main>
  );
}
