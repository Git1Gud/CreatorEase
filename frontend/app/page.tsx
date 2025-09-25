"use client"

import Link from "next/link"
import { Video, Clapperboard, Sparkles, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

const features = [
  {
    title: "Segment Clipper",
    description: "Generate AI-ranked highlights with ready-to-share captioned reels.",
    href: "/clipper",
    icon: Clapperboard,
    cta: "Generate clips",
  },
  {
    title: "Multicam Mixer",
    description: "Upload two camera angles plus clean audio to auto-sync and caption the final edit.",
    href: "/multicam",
    icon: Video,
    cta: "Open multicam",
  },
  // {
  //   title: "Segment Predictions",
  //   description: "Preview the top-performing hooks and download the generated assets instantly.",
  //   href: "/predictions",
  //   icon: Sparkles,
  //   cta: "Run predictions",
  // },
]

export default function HomePage() {
  return (
    <main className="container mx-auto px-4 py-16">
      <section className="mb-16 text-center">
        <span className="inline-flex items-center rounded-full border border-purple-500/50 bg-purple-500/10 px-3 py-1 text-xs uppercase tracking-wide text-purple-200">
          Creator toolkit
        </span>
        <h1 className="mt-6 text-4xl font-bold sm:text-5xl lg:text-6xl">
          Smart video tooling built for agile content teams
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-base text-gray-400 sm:text-lg">
          Choose the workflow you need: automate highlight discovery, sync multicam edits, or preview the highest-performing
          segments before you publish.
        </p>
        <div className="mt-10 flex justify-center gap-4">
          <Button asChild size="lg" className="bg-purple-600 hover:bg-purple-700">
            <Link href="/clipper" className="inline-flex items-center gap-2">
              Try the clipper
              <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
          <Button asChild size="lg" variant="outline" className="border-gray-700 text-gray-100 hover:bg-gray-900">
            <Link href="/multicam">Launch multicam</Link>
          </Button>
        </div>
      </section>

      <section className="grid gap-6 md:grid-cols-3">
        {features.map(({ title, description, href, icon: Icon, cta }) => (
          <Card
            key={title}
            className="group border-gray-800 bg-gray-900/60 backdrop-blur transition-colors hover:border-purple-500/60"
          >
            <CardHeader>
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-purple-500/20 text-purple-300">
                <Icon className="h-6 w-6" />
              </div>
              <CardTitle className="text-xl">{title}</CardTitle>
              <CardDescription className="text-gray-400">{description}</CardDescription>
            </CardHeader>
            <CardContent>
              <Button asChild className="w-full bg-purple-600 hover:bg-purple-700">
                <Link href={href} className="inline-flex items-center justify-center gap-2">
                  {cta}
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </CardContent>
          </Card>
        ))}
      </section>
    </main>
  )
}
