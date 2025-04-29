"use client"

import type React from "react"
import { useState } from "react"
import { useRouter } from "next/navigation"
import { Upload, Play, Loader2, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"

export default function VideoClipper() {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

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
      // In a real implementation, you would upload the file to the server
      // For this demo, we'll simulate the API call and response

      // Simulate progress while "uploading"
      setProgress(30)

      // Make API call to localhost:8000
      // In a real implementation, you would use FormData to upload the file
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:5000/process_video", {
        method: "POST",
        body: formData,
      });

      setProgress(70)

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`)
      }

      // For demo purposes, we'll use a hardcoded video URL
      console.log("Video processed successfully",response)
      const videoUrl = "https://zaidcre.s3.us-east-1.amazonaws.com/Screen+Recording+2025-04-23+235956.mp4"

      // Store the original video and generated clips in localStorage
      if (preview) {
        localStorage.setItem("originalVideo", preview)
      }

      // Store the clip URLs (using the same URL for all clips in this demo)
      localStorage.setItem("clip1", videoUrl)
      localStorage.setItem("clip2", videoUrl)
      localStorage.setItem("clip3", videoUrl)

      setProgress(100)

      // Navigate to results page
      router.push("/results")
    } catch (err) {
      console.error("Error processing video:", err)
      setError(err instanceof Error ? err.message : "Failed to process video. Please try again.")
      setIsProcessing(false)
    }
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-center bg-gradient-to-r from-purple-500 to-blue-500 bg-clip-text text-transparent">
          Video Clipper
        </h1>
        <p className="text-center text-gray-400 mt-2">Upload your video and get 3 optimized clips in return</p>
      </header>

      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-8 md:grid-cols-2">
        <Card className="bg-gray-900 border-gray-800">
          <CardContent className="p-6">
            <h2 className="text-xl font-semibold mb-4">Upload Video</h2>

            <div
              className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => document.getElementById("video-upload")?.click()}
            >
              <input type="file" id="video-upload" accept="video/*" className="hidden" onChange={handleFileChange} />
              <Upload className="mx-auto h-12 w-12 text-gray-500 mb-4" />
              <p className="text-gray-400 mb-2">Drag and drop your video here or click to browse</p>
              <p className="text-xs text-gray-500">Supports MP4, MOV, AVI (max 500MB)</p>
            </div>

            {file && (
              <div className="mt-4">
                <p className="text-sm text-gray-400 truncate">Selected: {file.name}</p>
                <p className="text-xs text-gray-500">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            )}

            <Button
              className="w-full mt-6 bg-blue-600 hover:bg-blue-700"
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

        <Card className="bg-gray-900 border-gray-800">
          <CardContent className="p-6">
            <h2 className="text-xl font-semibold mb-4">Preview</h2>

            {preview ? (
              <div className="aspect-video bg-black rounded-lg overflow-hidden">
                <video
                  src={preview}
                  controls
                  className="w-full h-full"
                  poster="/placeholder.svg?height=720&width=1280"
                />
              </div>
            ) : (
              <div className="aspect-video bg-gray-800 rounded-lg flex items-center justify-center">
                <Play className="h-16 w-16 text-gray-700" />
              </div>
            )}

            {isProcessing && (
              <div className="mt-6">
                <div className="flex justify-between text-sm mb-2">
                  <span>Processing video...</span>
                  <span>{progress}%</span>
                </div>
                <Progress value={progress} className="h-2 bg-gray-800" />
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
