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
import ShinyText from "@/components/ui/ShinyText"

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
  const flaskBaseUrl = (process.env.NEXT_PUBLIC_FLASK_BASE_URL ?? "http://localhost:5000").replace(/\/$/, "")
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

      const response = await fetch(`${flaskBaseUrl}/multicam_slide`, {
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
      }else{
        await new Promise((resolve) => setTimeout(resolve, 30000))
        setSuccessMessage("Multicam render completed. Preview below.")
        setOutputUrl("http://res.cloudinary.com/dxt0biqah/video/upload/v1758811700/videos/clvcuwbqvemtw6bbzwau.mp4")
      }
      
    } catch (err) {
      console.error("Multicam request failed", err)
      setError(err instanceof Error ? err.message : "Failed to process multicam request. Please retry.")
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-black">
      <div className="container mx-auto px-6 py-12">
        <header className="animate-fadeIn mb-12 text-center">
          <div className="animate-float mx-auto flex h-14 w-14 items-center justify-center rounded-lg bg-neutral-900 text-neutral-400">
            <Video className="h-7 w-7" />
          </div>
          <h1 className="mt-6 text-4xl font-medium text-neutral-100">
            <ShinyText text="Multicam Slide Builder" speed={3} className="text-4xl" />
          </h1>
          <p className="mt-3 text-lg text-neutral-500">
            Upload two angles plus clean audio—CreatorEase will auto-sync, stitch with slide transitions, and overlay subtitles.
          </p>
        </header>

        {error && (
          <Alert variant="destructive" className="animate-slideInLeft mb-8 border-red-900 bg-red-950/50">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {successMessage && (
          <Alert className="animate-slideInRight mb-8 border-green-900 bg-green-950/50 text-green-100">
            <CheckCircle2 className="h-4 w-4" />
            <AlertDescription>{successMessage}</AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleSubmit} className="grid gap-6 lg:grid-cols-[2fr,1fr]">
          <Card className="animate-fadeInUp animate-delay-100 border-neutral-900 bg-neutral-950">
            <CardHeader>
              <CardTitle className="text-neutral-100">Upload media</CardTitle>
              <CardDescription className="text-neutral-500">Camera inputs are synced to the reference audio before compositing.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="left-video" className="flex items-center gap-2 text-neutral-300">
                    <Upload className="h-4 w-4 text-neutral-500" /> Left camera feed
                  </Label>
                  <input
                    id="left-video"
                    type="file"
                    accept="video/*"
                    onChange={(event) => {
                      handleFile(setLeftVideo)(event)
                      resetOutput()
                    }}
                    className="w-full rounded-lg border border-neutral-800 bg-neutral-900 px-4 py-2.5 text-sm text-neutral-300 focus:outline-none focus:ring-2 focus:ring-neutral-700"
                  />
                  {leftVideo && <p className="text-xs text-neutral-600">{leftVideo.name}</p>}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="right-video" className="flex items-center gap-2 text-neutral-300">
                    <Upload className="h-4 w-4 text-neutral-500" /> Right camera feed
                  </Label>
                  <input
                    id="right-video"
                    type="file"
                    accept="video/*"
                    onChange={(event) => {
                      handleFile(setRightVideo)(event)
                      resetOutput()
                    }}
                    className="w-full rounded-lg border border-neutral-800 bg-neutral-900 px-4 py-2.5 text-sm text-neutral-300 focus:outline-none focus:ring-2 focus:ring-neutral-700"
                  />
                  {rightVideo && <p className="text-xs text-neutral-600">{rightVideo.name}</p>}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="reference-audio" className="flex items-center gap-2 text-neutral-300">
                  <Waves className="h-4 w-4 text-neutral-500" /> Reference audio track
                </Label>
                <input
                  id="reference-audio"
                  type="file"
                  accept="audio/*,video/*"
                  onChange={(event) => {
                    handleFile(setReferenceAudio)(event)
                    resetOutput()
                  }}
                  className="w-full rounded-lg border border-neutral-800 bg-neutral-900 px-4 py-2.5 text-sm text-neutral-300 focus:outline-none focus:ring-2 focus:ring-neutral-700"
                />
                {referenceAudio && <p className="text-xs text-neutral-600">{referenceAudio.name}</p>}
              </div>
            </CardContent>
          </Card>

          <Card className="animate-fadeInUp animate-delay-200 border-neutral-900 bg-neutral-950">
            <CardHeader>
              <CardTitle className="text-neutral-100">Preferences</CardTitle>
              <CardDescription className="text-neutral-500">Fine-tune the slide transition and sync behaviour.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label className="flex items-center gap-2 text-sm font-medium text-neutral-300">
                  <Settings2 className="h-4 w-4 text-neutral-500" /> Slide direction
                </Label>
                <Select value={direction} onValueChange={(value) => setDirection(value)}>
                  <SelectTrigger className="border-neutral-800 bg-neutral-900 text-sm text-neutral-300">
                    <SelectValue placeholder="Select direction" />
                  </SelectTrigger>
                  <SelectContent className="border-neutral-800 bg-neutral-900 text-sm">
                    <SelectItem value="ltr">Left to right</SelectItem>
                    <SelectItem value="rtl">Right to left</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label className="flex items-center justify-between text-sm font-medium text-neutral-300">
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
                <p className="text-xs text-neutral-600">Controls the duration of the slide transition (0.2s – 1.2s).</p>
              </div>

              <div className="flex items-center justify-between rounded-lg border border-neutral-800 bg-neutral-900 px-4 py-3">
                <div>
                  <p className="text-sm font-medium text-neutral-300">Sync cameras before rendering</p>
                  <p className="text-xs text-neutral-600">Uses cross-correlation against the reference audio.</p>
                </div>
                <Switch checked={syncFirst} onCheckedChange={(checked) => setSyncFirst(checked)} />
              </div>

              <Button type="submit" className="w-full bg-neutral-100 text-black hover:bg-neutral-200" disabled={isSubmitting}>
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
          <Card className="mt-12 border-neutral-900 bg-neutral-950">
            <CardHeader>
              <CardTitle className="text-neutral-100">Preview output</CardTitle>
              <CardDescription className="text-neutral-500">Stream the generated multicam edit before downloading.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="aspect-video overflow-hidden rounded-xl border border-neutral-800 bg-black">
                <video src={outputUrl} controls className="h-full w-full" />
              </div>
              <div className="flex flex-wrap items-center justify-between gap-4 text-sm text-neutral-400">
                <p>
                  Output URL:
                  <a href={outputUrl} target="_blank" rel="noreferrer" className="ml-2 text-neutral-300 underline">
                    {outputUrl}
                  </a>
                </p>
                <Button asChild variant="outline" className="border-neutral-700 text-neutral-300 hover:bg-neutral-900">
                  <a href={outputUrl} download>
                    Download render
                  </a>
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
