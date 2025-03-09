"use client"

import { useState } from "react"
import { Upload } from "lucide-react"
import DocumentUploader from "@/components/document-uploader"
import ChatInterface from "@/components/chat-interface"
import SummaryPanel from "@/components/summary-panel"
import { Button } from "@/components/ui/button"

export default function Home() {
  const [documentsUploaded, setDocumentsUploaded] = useState(false)
  const [summaries, setSummaries] = useState<string[]>([])
  const [finalSummary, setFinalSummary] = useState<string>("")
  const [keywords, setKeywords] = useState<string[][]>([])
  const [sentiment, setSentiment] = useState<{ label: string; score: number } | null>(null)

  const handleDocumentsProcessed = (
    docSummaries: string[],
    docKeywords: string[][],
    docSentiment: { label: string; score: number } | null,
    docFinalSummary: string,
  ) => {
    setSummaries(docSummaries)
    setKeywords(docKeywords)
    setSentiment(docSentiment)
    setFinalSummary(docFinalSummary)
    setDocumentsUploaded(true)
  }

  const resetApp = () => {
    setDocumentsUploaded(false)
    setSummaries([])
    setFinalSummary("")
    setKeywords([])
    setSentiment(null)
  }

  return (
    <main className="flex min-h-screen flex-col bg-background">
      <header className="border-b">
        <div className="container flex h-14 items-center px-4">
          <h1 className="text-xl font-bold">ML-Powered Research Assistant</h1>
          {documentsUploaded && (
            <Button variant="outline" size="sm" className="ml-auto" onClick={resetApp}>
              <Upload className="mr-2 h-4 w-4" />
              Upload New Documents
            </Button>
          )}
        </div>
      </header>

      <div className="flex-1 container flex flex-col md:flex-row">
        {!documentsUploaded ? (
          <div className="flex-1 flex items-center justify-center p-8">
            <DocumentUploader onDocumentsProcessed={handleDocumentsProcessed} />
          </div>
        ) : (
          <>
            <div className="flex-1 md:w-2/3 p-4 border-r">
              <ChatInterface documents={summaries} />
            </div>
            <div className="md:w-1/3 p-4">
              <SummaryPanel
                summaries={summaries}
                finalSummary={finalSummary}
                keywords={keywords}
                sentiment={sentiment}
              />
            </div>
          </>
        )}
      </div>
    </main>
  )
}

