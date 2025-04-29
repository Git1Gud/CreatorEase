"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft, Download, Maximize } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

export default function ResultsPage() {
  const router = useRouter()
  const [originalVideo, setOriginalVideo] = useState<string | null>(null)
  const [clips, setClips] = useState<{ url: string; name: string }[]>([])

  useEffect(() => {
    // Get the original video and clips from localStorage
    const savedOriginalVideo = localStorage.getItem("originalVideo")
    const clip1 = localStorage.getItem("clip1")
    const clip2 = localStorage.getItem("clip2")
    const clip3 = localStorage.getItem("clip3")

    // Default video URL in case localStorage is empty
    const defaultVideoUrl = "https://zaidcre.s3.us-east-1.amazonaws.com/Screen+Recording+2025-04-23+235956.mp4"

    if (savedOriginalVideo) {
      setOriginalVideo(savedOriginalVideo)
    } else {
      // If no video is found, redirect back to upload page
      router.push("/")
      return
    }

    // Set the clips with the retrieved URLs or fallback to the default URL
    setClips([
      {
        url: clip1 || defaultVideoUrl,
        name: "Clip_1.mp4",
      },
      {
        url: clip2 || defaultVideoUrl,
        name: "Clip_2.mp4",
      },
      {
        url: clip3 || defaultVideoUrl,
        name: "Clip_3.mp4",
      },
    ])
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
        <p className="text-gray-400 mt-2">Here are the 3 optimized clips generated from your video</p>
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
                  poster="/placeholder.svg?height=720&width=1280"
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

            <div className="grid gap-8 md:grid-cols-3">
              {clips.map((clip, index) => (
                <div key={index} className="space-y-4">
                  <div className="aspect-video bg-black rounded-lg overflow-hidden">
                    <video
                      id={`clip-${index}`}
                      src={clip.url}
                      className="w-full h-full"
                      controls
                      poster="/placeholder.svg?height=720&width=1280"
                    />
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
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
