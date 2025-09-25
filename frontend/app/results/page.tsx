"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft, Download, Maximize, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

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
    <div className="container mx-auto py-8 px-4">
      <header className="mb-8">
        <Button variant="ghost" className="mb-4 text-gray-400 hover:text-white" onClick={handleBackClick}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Upload
        </Button>

        <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-500 to-blue-500 bg-clip-text text-transparent">
          Your Generated Clips
        </h1>
        <p className="text-gray-400 mt-2 flex items-center justify-center gap-2">
          <Sparkles className="h-4 w-4 text-purple-300" />
          Here are the top-performing segments generated from your upload
        </p>
      </header>

      <div className="grid gap-8">
        <Card className="bg-gray-900 border-gray-800">
          <CardContent className="p-6">
            <h2 className="text-xl font-semibold mb-4">Original Video</h2>

            {originalVideo ? (
              <div className="aspect-video bg-black rounded-lg overflow-hidden">
                <video
                  src={originalVideo}
                  controls
                  className="w-full h-full"
                  // poster="/placeholder.svg?height=720&width=1280"
                />
              </div>
            ) : (
              <div className="aspect-video bg-gray-800 rounded-lg flex items-center justify-center">
                <p className="text-gray-500">Video not available</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="bg-gray-900 border-gray-800">
          <CardContent className="p-6">
            <h2 className="text-xl font-semibold mb-6">Generated Clips</h2>
            {clips.length ? (
              <div className="grid gap-8 md:grid-cols-3">
                {clips.map((clip, index) => (
                  <div key={index} className="space-y-4">
                    <div className="aspect-video bg-black rounded-lg overflow-hidden">
                      <video id={`clip-${index}`} src={clip.url} className="w-full h-full" controls />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Clip {index + 1}</span>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          className="gap-1"
                          onClick={() => handleFullscreen(`clip-${index}`)}
                        >
                          <Maximize className="h-4 w-4" />
                          Fullscreen
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="gap-1"
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
              <div className="rounded-lg border border-dashed border-gray-700 p-8 text-center text-gray-500">
                No clips available for download. Please process a video on the clipper page first.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
