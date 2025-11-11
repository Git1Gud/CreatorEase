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
  {
    title: "Video Editor",
    description: "Collaborate and edit video through text, detect silence and much more.",
    href: "http://localhost:5173/collaborate",
    icon: Sparkles,
    cta: "Open Editor",
    target: "_blank",
    rel: "noopener noreferrer",
  },
]

export default function HomePage() {
  return (
    <main className="min-h-screen bg-black">
      <div className="container mx-auto px-6 py-20">
        <section className="mb-20 text-center">
          <span className="animate-fadeIn inline-flex items-center rounded-full border border-neutral-800 bg-neutral-900 px-4 py-1.5 text-xs font-medium uppercase tracking-wider text-neutral-400">
            Creator toolkit
          </span>
          <h1 className="animate-fadeInUp animate-delay-100 mt-8 text-5xl font-medium tracking-tight text-neutral-100 sm:text-6xl lg:text-7xl">
            Smart video tooling built for
            <br />
            <span className="shiny-text">agile content teams</span>
          </h1>
          <p className="animate-fadeInUp animate-delay-200 mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-neutral-500">
            Choose the workflow you need: automate highlight discovery, sync multicam edits, or preview the highest-performing
            segments before you publish.
          </p>
          <div className="animate-fadeInUp animate-delay-300 mt-12 flex justify-center gap-4">
            <Button asChild size="lg" className="bg-neutral-100 text-black hover:bg-neutral-200">
              <Link href="/clipper" className="inline-flex items-center gap-2">
                Try the clipper
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline" className="border-neutral-800 text-neutral-300 hover:bg-neutral-900 hover:text-neutral-100">
              <Link href="/multicam">Launch multicam</Link>
            </Button>
          </div>
        </section>

        <section className="grid gap-5 md:grid-cols-3">
          {features.map(({ title, description, href, icon: Icon, cta, target, rel }, index) => (
            <Card
              key={title}
              className={`animate-scaleIn animate-delay-${(index + 4) * 100} glow-hover group border-neutral-900 bg-neutral-950 transition-all hover:border-neutral-800 hover:bg-neutral-900`}
            >
              <CardHeader className="space-y-4">
                <div className="animate-float inline-flex h-12 w-12 items-center justify-center rounded-lg bg-neutral-900 text-neutral-400 transition-colors group-hover:bg-neutral-800 group-hover:text-neutral-300">
                  <Icon className="h-6 w-6" />
                </div>
                <CardTitle className="text-xl font-medium text-neutral-100">{title}</CardTitle>
                <CardDescription className="text-sm leading-relaxed text-neutral-500">{description}</CardDescription>
              </CardHeader>
              <CardContent>
                <Button asChild className="w-full bg-neutral-900 text-neutral-100 hover:bg-neutral-800">
                  <Link href={href} target={target} rel={rel} className="inline-flex items-center justify-center gap-2">
                    {cta}
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>
          ))}
        </section>
      </div>
    </main>
  )
}
