import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    // Forward the request to the Flask backend
    const formData = await request.formData()

    const response = await fetch("http://localhost:5000/api/process", {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json()
      return NextResponse.json({ error: error.error || "Processing failed" }, { status: response.status })
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("API Error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
