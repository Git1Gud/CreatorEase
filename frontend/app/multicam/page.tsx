"use client"

import { useState } from "react"
import {
  Upload,
  Video,
  Waves,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Settings2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export default function MulticamPage() {
  const [leftVideo, setLeftVideo] = useState<File | null>(null)
  const [rightVideo, setRightVideo] = useState<File | null>(null)
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null)
  const [direction, setDirection] = useState("ltr")
  const [syncFirst, setSyncFirst] = useState(true)
  const [overlap, setOverlap] = useState<number[]>([60])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [outputUrl, setOutputUrl] = useState<string | null>(null)

  const handleFile = (setter: (file: File | null) => void) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0] ?? null
      setter(file)
      setError(null)
    }

  const resetOutput = () => {
    setSuccessMessage(null)
    setOutputUrl(null)
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (!leftVideo || !rightVideo || !referenceAudio) {
      setError("Please provide both camera feeds and the reference audio track.")
      return
    }

    setIsSubmitting(true)
    setError(null)
    setSuccessMessage(null)
    setOutputUrl(null)

    try {
      const formData = new FormData()
      formData.append("left_video", leftVideo)
      formData.append("right_video", rightVideo)
      formData.append("audio", referenceAudio)
      formData.append("direction", direction)
      formData.append("overlap", String(overlap[0] / 100))
      formData.append("sync_first", String(syncFirst))

      const response = await fetch("http://localhost:5000/multicam_slide", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Server responded with status ${response.status}`)
      }

      const data = await response.json()
      const output = typeof data?.output === "string" ? data.output : null
      setSuccessMessage(data?.message ?? "Multicam render completed. Preview below.")
      if (output) {
        setOutputUrl(output)
      }
    } catch (err) {
      console.error("Multicam request failed", err)
      setError(err instanceof Error ? err.message : "Failed to process multicam request. Please retry.")
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <header className="mb-10 text-center">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-purple-500/15 text-purple-300">
          <Video className="h-6 w-6" />
        </div>
        <h1 className="mt-4 text-3xl font-bold sm:text-4xl">Multicam Slide Builder</h1>
        <p className="mt-2 text-gray-400">
          Upload two angles plus clean audio—CreatorEase will auto-sync, stitch with slide transitions, and overlay subtitles.
        </p>
      </header>

      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {successMessage && (
        <Alert className="mb-6 border-green-500/40 bg-green-500/10 text-green-100">
          <CheckCircle2 className="h-4 w-4" />
          <AlertDescription>{successMessage}</AlertDescription>
        </Alert>
      )}

      <form onSubmit={handleSubmit} className="grid gap-8 lg:grid-cols-[2fr,1fr]">
        <Card className="border-gray-800 bg-gray-900">
          <CardHeader>
            <CardTitle>Upload media</CardTitle>
            <CardDescription>Camera inputs are synced to the reference audio before compositing.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="left-video" className="flex items-center gap-2">
                  <Upload className="h-4 w-4 text-purple-300" /> Left camera feed
                </Label>
                <input
                  id="left-video"
                  type="file"
                  accept="video/*"
                  onChange={(event) => {
                    handleFile(setLeftVideo)(event)
                    resetOutput()
                  }}
                  className="w-full rounded-md border border-gray-800 bg-gray-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                {leftVideo && <p className="text-xs text-gray-500">{leftVideo.name}</p>}
              </div>
              <div className="space-y-2">
                <Label htmlFor="right-video" className="flex items-center gap-2">
                  <Upload className="h-4 w-4 text-purple-300" /> Right camera feed
                </Label>
                <input
                  id="right-video"
                  type="file"
                  accept="video/*"
                  onChange={(event) => {
                    handleFile(setRightVideo)(event)
                    resetOutput()
                  }}
                  className="w-full rounded-md border border-gray-800 bg-gray-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                {rightVideo && <p className="text-xs text-gray-500">{rightVideo.name}</p>}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="reference-audio" className="flex items-center gap-2">
                <Waves className="h-4 w-4 text-purple-300" /> Reference audio track
              </Label>
              <input
                id="reference-audio"
                type="file"
                accept="audio/*,video/*"
                onChange={(event) => {
                  handleFile(setReferenceAudio)(event)
                  resetOutput()
                }}
                className="w-full rounded-md border border-gray-800 bg-gray-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              {referenceAudio && <p className="text-xs text-gray-500">{referenceAudio.name}</p>}
            </div>
          </CardContent>
        </Card>

        <Card className="border-gray-800 bg-gray-900">
          <CardHeader>
            <CardTitle>Preferences</CardTitle>
            <CardDescription>Fine-tune the slide transition and sync behaviour.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label className="flex items-center gap-2 text-sm font-medium">
                <Settings2 className="h-4 w-4 text-purple-300" /> Slide direction
              </Label>
              <Select value={direction} onValueChange={(value) => setDirection(value)}>
                <SelectTrigger className="bg-gray-950 border-gray-800 text-sm">
                  <SelectValue placeholder="Select direction" />
                </SelectTrigger>
                <SelectContent className="border-gray-800 bg-gray-900 text-sm">
                  <SelectItem value="ltr">Left to right</SelectItem>
                  <SelectItem value="rtl">Right to left</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="flex items-center justify-between text-sm font-medium">
                <span>Slide overlap ({(overlap[0] / 100).toFixed(2)}s)</span>
              </Label>
              <Slider
                value={overlap}
                min={20}
                max={120}
                step={5}
                onValueChange={(value) => {
                  setOverlap(value)
                  resetOutput()
                }}
                className="w-full"
              />
              <p className="text-xs text-gray-500">Controls the duration of the slide transition (0.2s – 1.2s).</p>
            </div>

            <div className="flex items-center justify-between rounded-md border border-gray-800 bg-gray-950 px-3 py-2">
              <div>
                <p className="text-sm font-medium">Sync cameras before rendering</p>
                <p className="text-xs text-gray-500">Uses cross-correlation against the reference audio.</p>
              </div>
              <Switch checked={syncFirst} onCheckedChange={(checked) => setSyncFirst(checked)} />
            </div>

            <Button type="submit" className="w-full bg-purple-600 hover:bg-purple-700" disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Rendering multicam...
                </>
              ) : (
                "Build multicam edit"
              )}
            </Button>
          </CardContent>
        </Card>
      </form>

      {outputUrl && (
        <Card className="mt-10 border-gray-800 bg-gray-900">
          <CardHeader>
            <CardTitle>Preview output</CardTitle>
            <CardDescription>Stream the generated multicam edit before downloading.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="aspect-video overflow-hidden rounded-lg border border-gray-800 bg-black">
              <video src={outputUrl} controls className="h-full w-full" />
            </div>
            <div className="flex flex-wrap items-center justify-between gap-4 text-sm text-gray-400">
              <p>
                Output URL:
                <a href={outputUrl} target="_blank" rel="noreferrer" className="ml-2 text-purple-300 underline">
                  {outputUrl}
                </a>
              </p>
              <Button asChild variant="outline" className="border-purple-500/40 text-purple-200 hover:bg-purple-500/10">
                <a href={outputUrl} download>
                  Download render
                </a>
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
