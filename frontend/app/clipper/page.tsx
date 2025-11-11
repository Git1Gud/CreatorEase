"use client"

import type React from "react"
import { useState } from "react"
import { useRouter } from "next/navigation"
import { Upload, Play, Loader2, AlertCircle, Clapperboard } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"

export default function ClipperPage() {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const flaskBaseUrl = (process.env.NEXT_PUBLIC_FLASK_BASE_URL ?? "http://localhost:5000").replace(/\/$/, "")
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile && selectedFile.type.startsWith("video/")) {
      setFile(selectedFile)
      const url = URL.createObjectURL(selectedFile)
      setPreview(url)
      setError(null)
    }
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && droppedFile.type.startsWith("video/")) {
      setFile(droppedFile)
      const url = URL.createObjectURL(droppedFile)
      setPreview(url)
      setError(null)
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }

  const processVideo = async () => {
    if (!file) return

    setIsProcessing(true)
    setProgress(10)
    setError(null)

    try {
      setProgress(30)

      const formData = new FormData()
      formData.append("file", file)

  const response = await fetch(`${flaskBaseUrl}/process_video`, {
        method: "POST",
        body: formData,
      })

      setProgress(70)

      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Server responded with status: ${response.status}`)
      }

      const data = await response.json()
      const clipUrls: string[] = Array.isArray(data?.urls) ? data.urls : []

      if (!clipUrls.length) {
        throw new Error("No clips generated. Please try again with a different video.")
      }

      if (preview) {
        localStorage.setItem("originalVideo", preview)
      }
      localStorage.setItem("clipUrls", JSON.stringify(clipUrls))

      setProgress(100)
      router.push("/results")
    } catch (err) {
      console.error("Error processing video:", err)
      setError(err instanceof Error ? err.message : "Failed to process video. Please try again.")
      setIsProcessing(false)
    }
  }

  return (
    <div className="min-h-screen bg-black">
      <div className="container mx-auto py-12 px-6">
        <header className="animate-fadeIn mb-12 text-center">
          <div className="animate-float mx-auto mb-6 flex h-14 w-14 items-center justify-center rounded-lg bg-neutral-900 text-neutral-400">
            <Clapperboard className="h-7 w-7" />
          </div>
          <h1 className="text-4xl font-medium text-neutral-100">
            <span className="gradient-text">Segment Clipper</span>
          </h1>
          <p className="text-neutral-500 mt-3 text-lg">Upload your long-form video and generate highlight clips with captioned hooks.</p>
        </header>

        {error && (
          <Alert variant="destructive" className="animate-slideInLeft mb-8 border-red-900 bg-red-950/50">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="grid gap-6 md:grid-cols-2">
          <Card className="animate-fadeInUp animate-delay-100 bg-neutral-950 border-neutral-900">
            <CardContent className="p-8">
              <h2 className="text-xl font-medium mb-6 text-neutral-100">Upload Video</h2>

              <div
                className="border-2 border-dashed border-neutral-800 rounded-xl p-12 text-center cursor-pointer hover:border-neutral-700 hover:bg-neutral-950 transition-all"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => document.getElementById("clipper-video-upload")?.click()}
              >
                <input
                  id="clipper-video-upload"
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={handleFileChange}
                />
                <Upload className="mx-auto h-12 w-12 text-neutral-700 mb-4" />
                <p className="text-neutral-400 mb-2">Drag and drop your video here or click to browse</p>
                <p className="text-xs text-neutral-600">Supports MP4, MOV, AVI (max 1GB)</p>
              </div>

              {file && (
                <div className="mt-6 p-4 bg-neutral-900 rounded-lg border border-neutral-800">
                  <p className="text-sm text-neutral-300 truncate font-medium">{file.name}</p>
                  <p className="text-xs text-neutral-600 mt-1">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
              )}

              <Button
                className="w-full mt-6 bg-neutral-100 text-black hover:bg-neutral-200"
                onClick={processVideo}
                disabled={!file || isProcessing}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  "Generate Clips"
                )}
            </Button>
          </CardContent>
        </Card>

        <Card className="bg-neutral-950 border-neutral-900">
          <CardContent className="p-8">
            <h2 className="text-xl font-medium mb-6 text-neutral-100">Preview</h2>

            {preview ? (
              <div className="aspect-video bg-black rounded-xl overflow-hidden border border-neutral-900">
                <video src={preview} controls className="w-full h-full" />
              </div>
            ) : (
              <div className="aspect-video bg-neutral-900 rounded-xl flex items-center justify-center border border-neutral-800">
                <Play className="h-16 w-16 text-neutral-800" />
              </div>
            )}

            {isProcessing && (
              <div className="mt-8 p-6 bg-neutral-900 rounded-xl border border-neutral-800">
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-neutral-400">Processing video...</span>
                  <span className="text-neutral-300 font-medium">{progress}%</span>
                </div>
                <Progress value={progress} className="h-2 bg-neutral-800" />
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      </div>
    </div>
  )
}
