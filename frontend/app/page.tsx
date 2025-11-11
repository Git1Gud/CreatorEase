"use client"

import Link from "next/link"
import { Video, Clapperboard, Sparkles, ArrowRight, Zap, Users, Clock, CheckCircle2, Play } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import ShinyText from "@/components/ui/ShinyText"
import GradientText from "@/components/ui/GradientText"
import BlurText from "@/components/ui/BlurText"

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

const benefits = [
  {
    icon: Zap,
    title: "Lightning Fast",
    description: "AI-powered processing that saves hours of manual work",
  },
  {
    icon: Users,
    title: "Collaborate",
    description: "Real-time editing and syncing across your team",
  },
  {
    icon: Clock,
    title: "Save Time",
    description: "Automate tedious tasks and focus on creativity",
  },
]

export default function HomePage() {
  return (
    <main className="min-h-screen bg-black">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-neutral-900/50 via-black to-black"></div>
        
        {/* Animated grid background */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:64px_64px]"></div>
        
        <div className="container relative mx-auto px-6 py-24 md:py-32 lg:py-40">
          <div className="mx-auto max-w-5xl text-center">
            {/* Badge */}
            <div className="animate-fadeIn mb-8">
              <span className="inline-flex items-center gap-2 rounded-full border border-neutral-800 bg-neutral-900/80 px-5 py-2.5 text-sm font-medium text-neutral-300 backdrop-blur-xl transition-colors hover:border-neutral-700 hover:bg-neutral-800/80">
                <Sparkles className="h-4 w-4 text-purple-400" />
                AI-Powered Video Platform
              </span>
            </div>
            
            {/* Main Heading */}
            <h1 className="animate-fadeInUp animate-delay-100 mb-8 text-5xl font-bold tracking-tight text-white sm:text-6xl md:text-7xl lg:text-8xl">
              Create videos at the
              <br />
              <ShinyText text="speed of thought" speed={3} className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold" />
            </h1>
            
            {/* Subtitle */}
            <BlurText
              text="Professional video editing powered by AI. Collaborate in real-time, generate clips instantly, and publish content faster than ever before."
              delay={50}
              animateBy="words"
              direction="top"
              className="mx-auto mb-12 max-w-3xl text-lg leading-relaxed text-neutral-400 md:text-xl text-center"
            />
            
            {/* CTA Buttons */}
            <div className="animate-fadeInUp animate-delay-300 flex flex-col items-center justify-center gap-4 sm:flex-row sm:gap-6">
              <Button asChild size="lg" className="group h-14 bg-white px-10 text-base font-semibold text-black transition-all hover:bg-neutral-100 hover:shadow-2xl hover:shadow-white/20">
                <Link href="/clipper" className="inline-flex items-center gap-3">
                  Get started free
                  <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="group h-14 border-2 border-neutral-700 bg-transparent px-10 text-base font-semibold text-white transition-all hover:border-neutral-500 hover:bg-neutral-900/50">
                <Link href="/multicam" className="inline-flex items-center gap-3">
                  <Play className="h-5 w-5" />
                  Watch demo
                </Link>
              </Button>
            </div>
          </div>
        </div>
        
        {/* Bottom fade */}
        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black to-transparent"></div>
      </div>

      {/* Benefits Section */}
      <div className="relative border-y border-neutral-900/50 py-24">
        <div className="container mx-auto px-6">
          <div className="mb-16 text-center">
            <h2 className="mb-4 text-3xl font-bold text-white md:text-4xl">
              <GradientText colors={['#ffffff', '#a3a3a3', '#ffffff']} animationSpeed={5}>
                Why creators choose CreatorEase
              </GradientText>
            </h2>
            <p className="mx-auto max-w-2xl text-lg text-neutral-500">
              Everything you need to create professional content, in one place
            </p>
          </div>
          
          <div className="grid gap-12 md:grid-cols-3">
            {benefits.map(({ icon: Icon, title, description }, index) => (
              <div
                key={title}
                className={`animate-fadeInUp animate-delay-${(index + 1) * 100} group text-center`}
              >
                <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-neutral-900 to-neutral-800 shadow-2xl transition-all duration-300 group-hover:scale-110 group-hover:from-neutral-800 group-hover:to-neutral-700 group-hover:shadow-neutral-800/50">
                  <Icon className="h-10 w-10 text-white transition-colors" />
                </div>
                <h3 className="mb-3 text-xl font-bold text-white">{title}</h3>
                <p className="text-base leading-relaxed text-neutral-500">{description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="relative py-32">
        <div className="container mx-auto px-6">
          <div className="mb-20 text-center">
            <h2 className="mb-6 text-4xl font-bold tracking-tight text-white md:text-5xl">
              Powerful tools for
              <br />
              <ShinyText text="modern creators" speed={4} className="text-4xl md:text-5xl font-bold" />
            </h2>
            <p className="mx-auto max-w-3xl text-xl text-neutral-400">
              From AI-powered clipping to real-time collaboration—everything you need to create viral content
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-3">
            {features.map(({ title, description, href, icon: Icon, cta, target, rel }, index) => (
              <Card
                key={title}
                className={`animate-scaleIn animate-delay-${(index + 4) * 100} group relative overflow-hidden border-2 border-neutral-800 bg-gradient-to-br from-neutral-950 to-neutral-900 backdrop-blur-xl transition-all duration-300 hover:border-neutral-600 hover:shadow-2xl hover:shadow-neutral-800/50`}
              >
                {/* Glow effect on hover */}
                <div className="absolute -inset-1 bg-gradient-to-r from-purple-600/20 to-blue-600/20 opacity-0 blur-xl transition-opacity duration-300 group-hover:opacity-100"></div>
                
                <CardHeader className="relative space-y-6 pb-6">
                  <div className="inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-neutral-800 to-neutral-900 p-4 shadow-lg transition-all duration-300 group-hover:scale-110 group-hover:from-white group-hover:to-neutral-200">
                    <Icon className="h-8 w-8 text-neutral-300 transition-colors group-hover:text-black" />
                  </div>
                  <div>
                    <CardTitle className="mb-3 text-2xl font-bold text-white">{title}</CardTitle>
                    <CardDescription className="text-base leading-relaxed text-neutral-400">
                      {description}
                    </CardDescription>
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <Button 
                    asChild 
                    className="group/btn w-full bg-white font-semibold text-black transition-all hover:bg-neutral-100 hover:shadow-xl"
                  >
                    <Link href={href} target={target} rel={rel} className="inline-flex items-center justify-center gap-2">
                      {cta}
                      <ArrowRight className="h-4 w-4 transition-transform group-hover/btn:translate-x-1" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="relative border-t border-neutral-900/50 py-32">
        <div className="container mx-auto px-6">
          <div className="relative mx-auto max-w-4xl overflow-hidden rounded-3xl border-2 border-neutral-800 bg-gradient-to-br from-neutral-950 via-neutral-900 to-neutral-950 p-12 text-center shadow-2xl md:p-16">
            {/* Animated gradient background */}
            <div className="absolute inset-0 bg-gradient-to-r from-purple-900/10 via-blue-900/10 to-purple-900/10 opacity-50"></div>
            
            <div className="relative">
              <h2 className="mb-6 text-4xl font-bold text-white md:text-5xl lg:text-6xl">
                Ready to transform your
                <br />
                <GradientText colors={['#ffffff', '#a3a3a3', '#ffffff']} animationSpeed={4}>
                  content workflow?
                </GradientText>
              </h2>
              <p className="mb-10 text-xl text-neutral-400">
                Join thousands of creators making better videos, faster
              </p>
              <Button asChild size="lg" className="group h-16 bg-white px-12 text-lg font-bold text-black transition-all hover:bg-neutral-100 hover:shadow-2xl hover:shadow-white/20">
                <Link href="/clipper" className="inline-flex items-center gap-3">
                  Start creating now
                  <ArrowRight className="h-6 w-6 transition-transform group-hover:translate-x-1" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
