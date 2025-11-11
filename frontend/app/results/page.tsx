"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft, Download, Maximize, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import GradientText from "@/components/ui/GradientText"

export default function ResultsPage() {
  const router = useRouter()
  const [originalVideo, setOriginalVideo] = useState<string | null>(null)
  const [clips, setClips] = useState<{ url: string; name: string }[]>([])

  useEffect(() => {
    // Get the original video and clips from localStorage
    const savedOriginalVideo = localStorage.getItem("originalVideo")
    let clipUrls: string[] = []
    const storedClips = localStorage.getItem("clipUrls")
    if (storedClips) {
      try {
        const parsed = JSON.parse(storedClips)
        clipUrls = Array.isArray(parsed) ? parsed : []
      } catch (parseError) {
        console.warn("Failed to parse clip URLs from storage", parseError)
      }
    }

    if (savedOriginalVideo) {
      setOriginalVideo(savedOriginalVideo)
    } else {
      // If no video is found, redirect back to upload page
      router.push("/")
      return
    }

    if (!clipUrls.length) {
      router.push("/clipper")
      return
    }

    setClips(
      clipUrls.map((url: string, index: number) => ({
        url,
        name: `Clip_${index + 1}.mp4`,
      }))
    )
  }, [router])

  const handleBackClick = () => {
    router.push("/")
  }

  const handleDownload = (url: string, filename: string) => {
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const handleFullscreen = (videoId: string) => {
    const videoElement = document.getElementById(videoId) as HTMLVideoElement
    if (videoElement) {
      if (videoElement.requestFullscreen) {
        videoElement.requestFullscreen()
      }
    }
  }

  return (
    <div className="min-h-screen bg-black">
      <div className="container mx-auto px-6 py-12">
        <header className="animate-fadeIn mb-12">
          <Button
            variant="ghost"
            className="mb-6 text-neutral-400 hover:bg-neutral-900 hover:text-neutral-100"
            onClick={handleBackClick}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Upload
          </Button>

          <div className="text-center">
            <div className="animate-float mx-auto flex h-14 w-14 items-center justify-center rounded-lg bg-neutral-900 text-neutral-400">
              <Sparkles className="h-7 w-7" />
            </div>
            <h1 className="mt-6 text-4xl font-medium text-neutral-100">
              <GradientText colors={['#ffffff', '#a3a3a3', '#ffffff']} animationSpeed={4}>
                Your Generated Clips
              </GradientText>
            </h1>
            <p className="mt-3 text-lg text-neutral-500">
              Top-performing segments ready to download
            </p>
          </div>
        </header>

        <div className="grid gap-8">
          <Card className="animate-fadeInUp animate-delay-100 border-neutral-900 bg-neutral-950">
            <CardContent className="p-8">
              <h2 className="mb-6 text-xl font-medium text-neutral-100">Original Video</h2>

              {originalVideo ? (
                <div className="aspect-video overflow-hidden rounded-xl border border-neutral-800 bg-black">
                  <video
                    src={originalVideo}
                    controls
                    className="h-full w-full"
                  />
                </div>
              ) : (
                <div className="flex aspect-video items-center justify-center rounded-xl bg-neutral-900">
                  <p className="text-neutral-600">Video not available</p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="animate-fadeInUp animate-delay-200 border-neutral-900 bg-neutral-950">
            <CardContent className="p-8">
              <h2 className="mb-8 text-xl font-medium text-neutral-100">Generated Clips</h2>
              {clips.length ? (
                <div className="grid gap-8 md:grid-cols-3">
                  {clips.map((clip, index) => (
                    <div key={index} className="space-y-4">
                      <div className="aspect-video overflow-hidden rounded-xl border border-neutral-800 bg-black">
                        <video id={`clip-${index}`} src={clip.url} className="h-full w-full" controls />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-neutral-300">Clip {index + 1}</span>
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            className="gap-1 border-neutral-700 text-neutral-400 hover:bg-neutral-900 hover:text-neutral-100"
                            onClick={() => handleFullscreen(`clip-${index}`)}
                          >
                            <Maximize className="h-4 w-4" />
                            Fullscreen
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="gap-1 border-neutral-700 text-neutral-400 hover:bg-neutral-900 hover:text-neutral-100"
                            onClick={() => handleDownload(clip.url, clip.name)}
                          >
                            <Download className="h-4 w-4" />
                            Download
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-dashed border-neutral-800 bg-neutral-900 p-8 text-center text-neutral-600">
                  No clips available for download. Please process a video on the clipper page first.
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
