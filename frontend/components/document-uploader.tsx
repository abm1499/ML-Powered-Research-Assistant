"use client";

import { useState, useRef } from "react";
import { Upload, FileText, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface DocumentUploaderProps {
  onDocumentsProcessed: (
    summaries: string[],
    keywords: string[][],
    sentiment: { label: string; score: number } | null,
    finalSummary: string,
  ) => void;
}

export default function DocumentUploader({ onDocumentsProcessed }: DocumentUploaderProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const fileArray = Array.from(e.target.files).filter((file) => file.type === "application/pdf");
      setFiles(fileArray);
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) return;

    setIsUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));

      console.log("Uploading files...");
      const uploadRes = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.statusText}`);
      console.log("Upload successful");


      for (let i = 0; i <= 100; i += 10) {
        setUploadProgress(i);
        await new Promise((resolve) => setTimeout(resolve, 300));
      }

      console.log("Fetching summaries...");
      const summariesRes = await fetch("http://localhost:8000/summaries");
      if (!summariesRes.ok) throw new Error(`Summaries failed: ${summariesRes.statusText}`);
      const { summaries } = await summariesRes.json();
      console.log("Summaries fetched:", summaries);

      let finalSummary = "";
      if (files.length > 1) {
        console.log("Fetching final summary...");
        const finalRes = await fetch("http://localhost:8000/final-summary");
        if (!finalRes.ok) throw new Error(`Final summary failed: ${finalRes.statusText}`);
        const { final_summary } = await finalRes.json();
        finalSummary = final_summary;
        console.log("Final summary fetched:", finalSummary);
      } else {
        finalSummary = summaries[0];
      }

      console.log("Fetching sentiment...");
      const sentimentRes = await fetch("http://localhost:8000/sentiment");
      if (!sentimentRes.ok) throw new Error(`Sentiment failed: ${sentimentRes.statusText}`);
      const { sentiment } = await sentimentRes.json();
      console.log("Sentiment fetched:", sentiment);

      console.log("Fetching keywords...");
      const keywordsRes = await fetch("http://localhost:8000/keywords");
      if (!keywordsRes.ok) throw new Error(`Keywords failed: ${keywordsRes.statusText}`);
      const { keywords } = await keywordsRes.json();
      console.log("Keywords fetched:", keywords);

      onDocumentsProcessed(summaries, keywords, sentiment, finalSummary);
    } catch (err) {
      console.error("Upload error:", err);
      setError(err instanceof Error ? err.message : "An unknown error occurred");
    } finally {
      setIsUploading(false);
    }
  };

  const handleOpenFileDialog = () => {
    console.log("Opening file dialog");
    fileInputRef.current?.click();
  };

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>Upload Research Papers</CardTitle>
        <CardDescription>Upload PDF documents to get started</CardDescription>
      </CardHeader>
      <CardContent>
        <div
          className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:bg-muted/50 transition-colors"
          onClick={handleOpenFileDialog}
        >
          <FileText className="h-10 w-10 mx-auto mb-4 text-muted-foreground" />
          <p className="text-sm text-muted-foreground mb-1">Click to upload or drag and drop</p>
          <p className="text-xs text-muted-foreground">PDF files only (max 10MB each)</p>
        </div>
        <input
          ref={fileInputRef}
          id="file-upload"
          type="file"
          multiple
          accept=".pdf"
          className="hidden"
          onChange={handleFileChange}
        />

        {files.length > 0 && (
          <div className="mt-4">
            <h4 className="text-sm font-medium mb-2">Selected files:</h4>
            <ul className="text-sm space-y-1">
              {files.map((file, index) => (
                <li key={index} className="flex items-center">
                  <FileText className="h-4 w-4 mr-2 text-primary" />
                  {file.name}
                </li>
              ))}
            </ul>
          </div>
        )}

        {isUploading && (
          <div className="mt-4 space-y-2">
            <Progress value={uploadProgress} className="h-2" />
            <p className="text-xs text-center text-muted-foreground">Processing documents... {uploadProgress}%</p>
          </div>
        )}

        {error && (
          <div className="mt-4 text-red-500 text-sm">
            Error: {error}
          </div>
        )}
      </CardContent>
      <CardFooter>
        <Button className="w-full" onClick={handleUpload} disabled={files.length === 0 || isUploading}>
          {isUploading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Processing
            </>
          ) : (
            <>
              <Upload className="mr-2 h-4 w-4" />
              Upload and Process
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}