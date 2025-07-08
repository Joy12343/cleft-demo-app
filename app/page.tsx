"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Upload, Download, Loader2, ImageIcon } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

export default function FaceInpaintingApp() {
  const [sourceImage, setSourceImage] = useState<File | null>(null)
  const [maskImage, setMaskImage] = useState<File | null>(null)
  const [sourcePreview, setSourcePreview] = useState<string | null>(null)
  const [maskPreview, setMaskPreview] = useState<string | null>(null)
  const [resultImage, setResultImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [landmarkImage, setLandmarkImage] = useState<string | null>(null)

  const sourceInputRef = useRef<HTMLInputElement>(null)
  const maskInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()

  const handleImageUpload = (file: File, type: "source" | "mask") => {
    if (!file.type.startsWith("image/")) {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file",
        variant: "destructive",
      })
      return
    }

    const reader = new FileReader()
    reader.onload = (e) => {
      const result = e.target?.result as string
      if (type === "source") {
        setSourceImage(file)
        setSourcePreview(result)
      } else {
        setMaskImage(file)
        setMaskPreview(result)
      }
    }
    reader.readAsDataURL(file)
  }

  const processImages = async () => {
    if (!sourceImage || !maskImage) {
      toast({
        title: "Missing images",
        description: "Please upload both source and mask images",
        variant: "destructive",
      })
      return
    }

    setIsProcessing(true)
    setResultImage(null)
    setLandmarkImage(null)

    try {
      const formData = new FormData()
      formData.append("source", sourceImage)
      formData.append("mask", maskImage)

      const response = await fetch("/api/process", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || "Processing failed")
      }

      const data = await response.json()
      setResultImage(data.result_url)
      setLandmarkImage(data.landmark_url)

      toast({
        title: "Processing complete!",
        description: "Your image has been successfully processed",
      })
    } catch (error) {
      console.error("Processing error:", error)
      toast({
        title: "Processing failed",
        description: error instanceof Error ? error.message : "An error occurred during processing",
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
    }
  }

  const downloadImage = (url: string, filename: string) => {
    const link = document.createElement("a")
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const resetAll = () => {
    setSourceImage(null)
    setMaskImage(null)
    setSourcePreview(null)
    setMaskPreview(null)
    setResultImage(null)
    setLandmarkImage(null)
    if (sourceInputRef.current) sourceInputRef.current.value = ""
    if (maskInputRef.current) maskInputRef.current.value = ""
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Face Inpainting Tool</h1>
          <p className="text-lg text-gray-600">Upload your source image and mask to generate inpainted results</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Images
              </CardTitle>
              <CardDescription>Upload your source photo and the painted white mask</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Source Image Upload */}
              <div className="space-y-2">
                <Label htmlFor="source-upload">Source Photo</Label>
                <Input
                  id="source-upload"
                  type="file"
                  accept="image/*"
                  ref={sourceInputRef}
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleImageUpload(file, "source")
                  }}
                />
                {sourcePreview && (
                  <div className="mt-2">
                    <img
                      src={sourcePreview || "/placeholder.svg"}
                      alt="Source preview"
                      className="w-full h-48 object-cover rounded-lg border"
                    />
                  </div>
                )}
              </div>

              {/* Mask Image Upload */}
              <div className="space-y-2">
                <Label htmlFor="mask-upload">Painted White Mask</Label>
                <Input
                  id="mask-upload"
                  type="file"
                  accept="image/*"
                  ref={maskInputRef}
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleImageUpload(file, "mask")
                  }}
                />
                {maskPreview && (
                  <div className="mt-2">
                    <img
                      src={maskPreview || "/placeholder.svg"}
                      alt="Mask preview"
                      className="w-full h-48 object-cover rounded-lg border"
                    />
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2">
                <Button
                  onClick={processImages}
                  disabled={!sourceImage || !maskImage || isProcessing}
                  className="flex-1"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <ImageIcon className="w-4 h-4 mr-2" />
                      Process Images
                    </>
                  )}
                </Button>
                <Button variant="outline" onClick={resetAll}>
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Results Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Download className="w-5 h-5" />
                Results
              </CardTitle>
              <CardDescription>Download your processed images</CardDescription>
            </CardHeader>
            <CardContent>
              {isProcessing && (
                <div className="flex items-center justify-center h-48 bg-gray-50 rounded-lg">
                  <div className="text-center">
                    <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2 text-blue-500" />
                    <p className="text-gray-600">Processing your images...</p>
                  </div>
                </div>
              )}

              {resultImage && (
                <div className="space-y-4">
                  <div>
                    <h3 className="font-semibold mb-2">Inpainted Result</h3>
                    <img
                      src={resultImage || "/placeholder.svg"}
                      alt="Processed result"
                      className="w-full h-48 object-cover rounded-lg border mb-2"
                    />
                    <Button
                      onClick={() => downloadImage(resultImage, "inpainted_result.jpg")}
                      variant="outline"
                      size="sm"
                      className="w-full"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download Result
                    </Button>
                  </div>

                  {landmarkImage && (
                    <div>
                      <h3 className="font-semibold mb-2">Detected Landmarks</h3>
                      <img
                        src={landmarkImage || "/placeholder.svg"}
                        alt="Detected landmarks"
                        className="w-full h-48 object-cover rounded-lg border mb-2"
                      />
                      <Button
                        onClick={() => downloadImage(landmarkImage, "landmarks.jpg")}
                        variant="outline"
                        size="sm"
                        className="w-full"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download Landmarks
                      </Button>
                    </div>
                  )}
                </div>
              )}

              {!isProcessing && !resultImage && (
                <div className="flex items-center justify-center h-48 bg-gray-50 rounded-lg">
                  <p className="text-gray-500">Upload images and click process to see results</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Instructions */}
        <Card>
          <CardHeader>
            <CardTitle>How to Use</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="list-decimal list-inside space-y-2 text-gray-700">
              <li>Upload your source photo containing the face you want to inpaint</li>
              <li>Upload a mask image where the areas to be inpainted are painted white</li>
              <li>Click "Process Images" to generate the inpainted result</li>
              <li>Download the processed result and landmark detection images</li>
            </ol>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
