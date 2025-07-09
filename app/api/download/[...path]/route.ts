import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  try {
    const { path } = await params
    const sessionId = path[0]
    const filename = path[1]

    const response = await fetch(`http://localhost:5000/api/download/${sessionId}/${filename}`)

    if (!response.ok) {
      return NextResponse.json({ error: "File not found" }, { status: 404 })
    }

    const buffer = await response.arrayBuffer()

    return new NextResponse(buffer, {
      headers: {
        "Content-Type": response.headers.get("Content-Type") || "application/octet-stream",
        "Content-Disposition": `attachment; filename="${filename}"`,
      },
    })
  } catch (error) {
    console.error("Download Error:", error)
    return NextResponse.json({ error: "Download failed" }, { status: 500 })
  }
}
