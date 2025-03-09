"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Paperclip, Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Avatar } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

interface ChatInterfaceProps {
  documents: string[];
}

export default function ChatInterface({ documents }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content: "Hello! I've analyzed your documents. Ask questions about them.",
      role: "assistant",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    const res = await fetch("http://localhost:8000/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: input }),
    });
    const { answer } = await res.json();

    const botResponse: Message = {
      id: (Date.now() + 1).toString(),
      content: answer,
      role: "assistant",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, botResponse]);
    setIsLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="text-xl font-semibold mb-4">Ask Questions</div>

      <Card className="flex-1 flex flex-col overflow-hidden">
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn("flex items-start gap-3 max-w-[80%]", message.role === "user" ? "ml-auto justify-end" : "")}
              >
                {message.role === "assistant" && (
                  <Avatar className="h-8 w-8">
                    <Bot className="h-5 w-5" />
                  </Avatar>
                )}
                <div
                  className={cn(
                    "rounded-lg p-3",
                    message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted",
                  )}
                >
                  <ReactMarkdown>{message.content}</ReactMarkdown> {/* Render with markdown */}
                </div>
                {message.role === "user" && (
                  <Avatar className="h-8 w-8 bg-primary">
                    <div className="text-xs text-primary-foreground">You</div>
                  </Avatar>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex items-start gap-3 max-w-[80%]">
                <Avatar className="h-8 w-8">
                  <Bot className="h-5 w-5" />
                </Avatar>
                <div className="rounded-lg p-3 bg-muted">
                  <div className="flex space-x-2">
                    <div className="h-2 w-2 rounded-full bg-muted-foreground/40 animate-bounce" />
                    <div
                      className="h-2 w-2 rounded-full bg-muted-foreground/40 animate-bounce"
                      style={{ animationDelay: "0.2s" }}
                    />
                    <div
                      className="h-2 w-2 rounded-full bg-muted-foreground/40 animate-bounce"
                      style={{ animationDelay: "0.4s" }}
                    />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        <div className="p-4 border-t">
          <div className="flex items-end gap-2">
            <Textarea
              placeholder="Ask a question about your documents..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              className="min-h-[60px] resize-none"
            />
            <div className="flex flex-col gap-2">
              <Button size="icon" variant="outline" type="button" className="rounded-full">
                <Paperclip className="h-4 w-4" />
                <span className="sr-only">Attach file</span>
              </Button>
              <Button
                size="icon"
                type="button"
                className="rounded-full"
                onClick={handleSendMessage}
                disabled={!input.trim() || isLoading}
              >
                <Send className="h-4 w-4" />
                <span className="sr-only">Send message</span>
              </Button>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}