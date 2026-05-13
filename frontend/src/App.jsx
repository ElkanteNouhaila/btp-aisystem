import { useState, useEffect, useCallback } from "react";
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";
const api = axios.create({
  baseURL: API_BASE_URL,
});

function App() {
  const [file, setFile] = useState(null);
  const [docs, setDocs] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [uploadType, setUploadType] = useState("pdf");


  const fetchDocs = useCallback(async () => {
    try {
      const res = await api.get("/docs");
      setDocs(Array.isArray(res.data) ? res.data : []);
      setError("");
    } catch {
      setError("Cannot connect to backend. Make sure Flask is running on port 5000.");
    }
  }, []);
  
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchDocs();
  }, [fetchDocs]);
  
const uploadFile = async () => {
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const endpoint = uploadType === "email" ? "/upload-email" : "/upload";

  try {
    setIsUploading(true);

    await api.post(endpoint, formData);

    await fetchDocs();

    setError("");
  } catch {
    setError("Upload failed. Check backend connection and file format.");
  } finally {
    setIsUploading(false);
  }
};
  const askQuestion = async () => {
    try {
      setIsAsking(true);
      const payload = {
        question,
        doc_id: selectedDoc || null,
      type: uploadType 
      };
      if (selectedDoc) {
        payload.doc_id = selectedDoc;
      }
      const res = await api.post("/ask", payload);
      setAnswer(res.data.answer || "");
      setError("");
    } catch {
      setError("Question request failed. Check backend connection.");
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f5f8ff] p-4 sm:p-8">
      <div className="mx-auto w-full max-w-5xl overflow-hidden rounded-3xl border border-[#dfe5ff] bg-white shadow-[0_20px_60px_-24px_rgba(29,42,103,0.45)]">
        <div className="bg-[#1D2A67] px-6 py-8 text-white sm:px-10">
          <p className="text-sm font-medium text-[#76C043]">RAG Assistant</p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">
            BTP AI System
          </h1>
          <p className="mt-2 text-sm text-[#d9e1ff]">
            Upload documents, select scope, and ask focused questions.
          </p>
        </div>

        <div className="grid gap-6 p-6 sm:grid-cols-2 sm:p-8">
          <section className="space-y-5 rounded-2xl border border-[#e9eeff] bg-[#f9fbff] p-5">
            <h2 className="text-lg font-semibold text-[#1D2A67]">Document Setup</h2>

            <div className="space-y-2">
              <select
                value={uploadType}
                onChange={(e) => setUploadType(e.target.value)}
                className="w-full rounded-xl border border-[#c9d4ff] bg-white px-3 py-2 text-sm text-[#1D2A67]"
              >
                <option value="pdf">PDF</option>
                <option value="email">Email</option>
              </select>
              <label className="text-sm font-medium text-[#1D2A67]">Upload File</label>
              <input
                type="file"
                onChange={(e) => setFile(e.target.files[0])}
                className="block w-full rounded-xl border border-[#c9d4ff] bg-white px-3 py-2 text-sm text-[#1D2A67] file:mr-4 file:rounded-lg file:border-0 file:bg-[#1D2A67] file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-[#162150]"
              />
              <button
                onClick={uploadFile}
                disabled={!file || isUploading}
                className="w-full rounded-xl bg-[#1D2A67] px-4 py-2.5 font-medium text-white transition hover:bg-[#162150] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isUploading ? "Uploading..." : "Upload Document"}
              </button>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-[#1D2A67]">Document Scope</label>
              <select
                value={selectedDoc}
                onChange={(e) => setSelectedDoc(e.target.value)}
                className="w-full rounded-xl border border-[#c9d4ff] bg-white px-3 py-2 text-sm text-[#1D2A67] outline-none focus:border-[#76C043] focus:ring-2 focus:ring-[#76C043]/40"
              >
                <option value="">All documents</option>
                {docs.map((d, i) => (
                  <option key={i} value={d.doc_id}>
                    {d.filename}
                  </option>
                ))}
              </select>
            </div>
          </section>

          <section className="space-y-5 rounded-2xl border border-[#e9eeff] bg-white p-5">
            <h2 className="text-lg font-semibold text-[#1D2A67]">Ask Question</h2>

            <textarea
              placeholder="Type your question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={5}
              className="w-full resize-none rounded-xl border border-[#c9d4ff] px-3 py-2 text-sm text-[#1D2A67] outline-none focus:border-[#76C043] focus:ring-2 focus:ring-[#76C043]/40"
            />

            <button
              onClick={askQuestion}
              disabled={!question.trim() || isAsking}
              className="w-full rounded-xl bg-[#76C043] px-4 py-2.5 font-semibold text-white transition hover:bg-[#69ab3b] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isAsking ? "Generating answer..." : "Ask AI"}
            </button>
          </section>
        </div>

        {error ? (
          <div className="mx-6 mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 sm:mx-8">
            {error}
          </div>
        ) : null}

        <section className="mx-6 mb-8 rounded-2xl border border-[#dbe4ff] bg-[#f7f9ff] p-5 sm:mx-8">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-[#1D2A67]">Answer</h3>
            <span className="rounded-full bg-[#1D2A67]/10 px-3 py-1 text-xs font-semibold text-[#1D2A67]">
              {selectedDoc ? "Filtered document" : "Global search"}
            </span>
          </div>
          <p className="min-h-[96px] whitespace-pre-line leading-relaxed text-[#24357f]">
            {answer || "No answer yet..."}
          </p>
        </section>
      </div>
    </div>
  );
}

export default App;

