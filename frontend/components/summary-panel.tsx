"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import ReactMarkdown from "react-markdown";

interface SummaryPanelProps {
  summaries: string[];
  finalSummary: string;
  keywords: string[][];
  sentiment: { label: string; score: number } | null;
}

export default function SummaryPanel({ summaries, finalSummary, keywords, sentiment }: SummaryPanelProps) {
  const [activeTab, setActiveTab] = useState("summaries");

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col overflow-hidden">
      <div className="text-xl font-semibold mb-4">Analysis Results</div>

      <Tabs defaultValue="summaries" className="flex-1 flex flex-col overflow-hidden" value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4">
          <TabsTrigger value="summaries">Summaries</TabsTrigger>
          <TabsTrigger value="keywords">Keywords</TabsTrigger>
          <TabsTrigger value="sentiment">Sentiment</TabsTrigger>
          <TabsTrigger value="final">Final</TabsTrigger>
        </TabsList>

        <TabsContent value="summaries" className="flex-1 overflow-hidden">
          <ScrollArea className="h-full pr-4" style={{ maxHeight: "calc(100% - 2rem)" }}>
            <div className="space-y-4">
              {summaries.map((summary, index) => (
                <Card key={index} className="mb-4">
                  <CardHeader className="py-3">
                    <CardTitle className="text-sm">Document {index + 1} Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ReactMarkdown>{summary}</ReactMarkdown>
                  </CardContent>
                </Card>
              ))}
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="keywords" className="flex-1 overflow-hidden">
          <ScrollArea className="h-full pr-4" style={{ maxHeight: "calc(100% - 2rem)" }}>
            <div className="space-y-4">
              {keywords.map((docKeywords, index) => (
                <Card key={index} className="mb-4">
                  <CardHeader className="py-3">
                    <CardTitle className="text-sm">Document {index + 1} Keywords</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {docKeywords.map((keyword, kidx) => (
                        <Badge key={kidx} variant="secondary">
                          {keyword}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="sentiment" className="flex-1">
          {sentiment && (
            <Card className="h-full">
              <CardHeader>
                <CardTitle>Sentiment Analysis</CardTitle>
                <CardDescription>Overall sentiment across all documents</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col items-center justify-center h-[300px]">
                  <div
                    className={`text-4xl font-bold mb-4 ${
                      sentiment.label === "POSITIVE"
                        ? "text-green-500"
                        : sentiment.label === "NEGATIVE"
                          ? "text-red-500"
                          : "text-yellow-500"
                    }`}
                  >
                    {sentiment.label}
                  </div>
                  <div className="w-full max-w-md h-4 bg-muted rounded-full overflow-hidden">
                    <div
                      className={`h-full ${
                        sentiment.label === "POSITIVE"
                          ? "bg-green-500"
                          : sentiment.label === "NEGATIVE"
                            ? "bg-red-500"
                            : "bg-yellow-500"
                      }`}
                      style={{ width: `${sentiment.score * 100}%` }}
                    />
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Confidence: {(sentiment.score * 100).toFixed(1)}%
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="final" className="flex-1 overflow-hidden">
          <Card className="h-full flex flex-col">
            <CardHeader>
              <CardTitle>Final Summary</CardTitle>
              <CardDescription>Comprehensive analysis across all documents</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden">
              <ScrollArea
                className="h-full pr-4"
                style={{ maxHeight: "100%", overflowY: "auto" }}
              >
                <div className="min-h-full">
                  <ReactMarkdown>{finalSummary}</ReactMarkdown>
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}