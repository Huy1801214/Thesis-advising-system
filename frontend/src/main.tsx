import React, { FormEvent, useEffect, useMemo, useState } from "react";
import ReactDOM from "react-dom/client";
import {
  BookOpenCheck,
  GraduationCap,
  LogIn,
  LogOut,
  LockKeyhole,
  Mail,
  MessageSquareText,
  Send,
  ShieldCheck,
  Sparkles,
  UserPlus,
  UserRound,
  Upload,
  FileSpreadsheet,
  Loader2,
  ChevronDown,
} from "lucide-react";
import { login, register, sendQuestion, uploadTranscript, ChatTraceItem, sendQuestionStream } from "./api";
import "./styles.css";

type ThinkingStep = {
  id: string;
  label: string;
  status: "pending" | "running" | "completed" | "failed";
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  criticScore?: number;
  debugTrace?: ChatTraceItem[];
  thinkingSteps?: ThinkingStep[];
  isClarify?: boolean;
};

const TOKEN_STORAGE_KEY = "thesis_advising_token";
type AuthMode = "login" | "register";

function createId() {
  return crypto.randomUUID?.() || `msg_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function formatResult(result: unknown) {
  if (typeof result === "string") return result;
  if (result === null || result === undefined) return "";
  return JSON.stringify(result, null, 2);
}

function App() {
  const inputRef = React.useRef<HTMLTextAreaElement>(null);
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_STORAGE_KEY));
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [mssv, setMssv] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loginError, setLoginError] = useState("");
  const [registerMessage, setRegisterMessage] = useState("");
  const [chatError, setChatError] = useState("");
  const [question, setQuestion] = useState("");
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [activeThinkingSteps, setActiveThinkingSteps] = useState<ThinkingStep[]>([]);
  const [openThinkingMsgs, setOpenThinkingMsgs] = useState<{ [msgId: string]: boolean }>({});
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccessMessage, setUploadSuccessMessage] = useState("");
  const [uploadErrorMessage, setUploadErrorMessage] = useState("");
  const [extractedGpa, setExtractedGpa] = useState<string | null>(null);
  const [extractedPassedCount, setExtractedPassedCount] = useState<number | null>(null);

  function toggleThinking(msgId: string) {
    setOpenThinkingMsgs((prev) => ({
      ...prev,
      [msgId]: !prev[msgId],
    }));
  }
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Chào mừng bạn đến với hệ thống tư vấn khóa luận.",
      sources: [
        "Hãy đặt câu hỏi về đề tài, quy trình, điều kiện thực hiện hoặc thông tin liên quan đến khóa luận.",
      ],
    },
  ]);

  const canSubmitQuestion = useMemo(
    () => question.trim().length > 0 && !isAsking,
    [isAsking, question],
  );

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoginError("");
    setRegisterMessage("");
    setIsLoggingIn(true);

    try {
      const result = await login(mssv.trim(), password);
      localStorage.setItem(TOKEN_STORAGE_KEY, result.access_token);
      setToken(result.access_token);
      setPassword("");
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : "Đăng nhập không thành công.");
    } finally {
      setIsLoggingIn(false);
    }
  }

  async function handleRegister(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoginError("");
    setRegisterMessage("");

    if (password !== confirmPassword) {
      setLoginError("Mật khẩu xác nhận không khớp.");
      return;
    }

    setIsLoggingIn(true);

    try {
      await register(mssv.trim(), email.trim(), password);
      const result = await login(mssv.trim(), password);
      localStorage.setItem(TOKEN_STORAGE_KEY, result.access_token);
      setToken(result.access_token);
      setRegisterMessage("Tài khoản đã được tạo thành công.");
      setPassword("");
      setConfirmPassword("");
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : "Đăng kí không thành công.");
    } finally {
      setIsLoggingIn(false);
    }
  }

  function switchAuthMode(nextMode: AuthMode) {
    setAuthMode(nextMode);
    setLoginError("");
    setRegisterMessage("");
    setPassword("");
    setConfirmPassword("");
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token || !canSubmitQuestion) return;

    const prompt = question.trim();
    setMessages((current) => [
      ...current,
      {
        id: createId(),
        role: "user",
        content: prompt,
      },
    ]);
    setQuestion("");
    setChatError("");
    setIsAsking(true);
    setActiveThinkingSteps([]);

    let finalAns = "";
    let criticScr = 0;
    let debugTrc: ChatTraceItem[] = [];

    try {
      await sendQuestionStream(prompt, token, (event, data) => {
        if (event === "planner_start") {
          setActiveThinkingSteps([
            { id: "planner", label: "Lập kế hoạch truy vấn (Planner)", status: "running" }
          ]);
        }
        else if (event === "planner_done") {
          setActiveThinkingSteps((prev) => {
            const updated = prev.map(s => s.id === "planner" ? { ...s, status: "completed" as const } : s);
            const subtasks = data.tasks.map((t: any) => ({
              id: t.task_id,
              label: `Chạy Agent ${t.task_type}: "${t.query_intent}"`,
              status: "pending" as const
            }));
            return [...updated, ...subtasks];
          });
        }
        else if (event === "executor_running") {
          setActiveThinkingSteps((prev) =>
            prev.map(s => data.running_tasks.includes(s.id) ? { ...s, status: "running" as const } : s)
          );
        }
        else if (event === "task_completed") {
          setActiveThinkingSteps((prev) =>
            prev.map(s => s.id === data.task.task_id ? { ...s, status: "completed" as const } : s)
          );
        }
        else if (event === "synthesizer_start") {
          setActiveThinkingSteps((prev) => [
            ...prev,
            { id: "synthesizer", label: "Tổng hợp thông tin câu trả lời (Synthesizer)", status: "running" as const }
          ]);
        }
        else if (event === "synthesizer_done") {
          setActiveThinkingSteps((prev) =>
            prev.map(s => s.id === "synthesizer" ? { ...s, status: "completed" as const } : s)
          );
        }
        else if (event === "critic_start") {
          setActiveThinkingSteps((prev) => [
            ...prev,
            { id: "critic", label: "Critic Agent phản biện và kiểm duyệt chất lượng", status: "running" as const }
          ]);
        }
        else if (event === "critic_done") {
          setActiveThinkingSteps((prev) =>
            prev.map(s => s.id === "critic" ? { ...s, status: "completed" as const } : s)
          );
        }
        else if (event === "final_result") {
          finalAns = data.answer;
          criticScr = data.critic_score;
          debugTrc = data.debug_trace;

          const requiresClarification = data.debug_trace?.some((t: any) => t.status === "CLARIFYING");
          if (requiresClarification) {
            setTimeout(() => inputRef.current?.focus(), 100);
          }
        }
      });

      // Ghi nhận tin nhắn cuối cùng kèm timeline suy nghĩ đã lưu
      setActiveThinkingSteps((prev) => {
        const finalized = prev.map(s => 
          s.status === "running" || s.status === "pending" ? { ...s, status: "completed" as const } : s
        );
        
        const isClarify = debugTrc?.some((t: any) => t.status === "CLARIFYING");

        setMessages((current) => [
          ...current,
          {
            id: `msg_${Date.now()}`,
            role: "assistant",
            content: finalAns || "Chưa tìm thấy kết quả phù hợp.",
            criticScore: criticScr,
            debugTrace: debugTrc,
            thinkingSteps: finalized,
            isClarify: isClarify
          },
        ]);
        return [];
      });

    } catch (error) {
      const message = error instanceof Error ? error.message : "Không thể gửi câu hỏi.";
      setChatError(message);

      if (message.toLowerCase().includes("đăng nhập")) {
        handleLogout();
      }
    } finally {
      setIsAsking(false);
      setActiveThinkingSteps([]);
    }
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    setUploadSuccessMessage("");
    setUploadErrorMessage("");
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  }

  async function handleUploadTranscript() {
    if (!selectedFile || !token) return;
    setIsUploading(true);
    setUploadSuccessMessage("");
    setUploadErrorMessage("");

    try {
      const result = await uploadTranscript(selectedFile, token);
      setUploadSuccessMessage(result.message || "Đồng bộ bảng điểm thành công!");
      if (result.data) {
        setExtractedGpa(result.data.gpa);
        setExtractedPassedCount(result.data.total_passed);
      }
      setSelectedFile(null);
    } catch (error) {
      setUploadErrorMessage(error instanceof Error ? error.message : "Tải lên bảng điểm thất bại.");
    } finally {
      setIsUploading(false);
    }
  }

  function handleLogout() {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    setToken(null);
    setChatError("");
    setSelectedFile(null);
    setUploadSuccessMessage("");
    setUploadErrorMessage("");
    setExtractedGpa(null);
    setExtractedPassedCount(null);
    setActiveThinkingSteps([]);
    setOpenThinkingMsgs({});
  }

  if (!token) {
    return (
      <main className="login-shell">
        <section className="login-hero" aria-labelledby="login-title">
          <div className="mark">
            <GraduationCap size={34} />
          </div>
          <p className="kicker">Thesis Advising System</p>
          <h1 id="login-title">Tư vấn khóa luận thông minh cho sinh viên.</h1>
          <p className="hero-copy">
            Đăng nhập bằng MSSV để hỏi đáp với hệ thống RAG và tri thức đồ thị của khoa.
          </p>
          <div className="hero-stats" aria-label="Tinh nang chinh">
            <span>
              <BookOpenCheck size={18} /> RAG
            </span>
            <span>
              <ShieldCheck size={18} /> Secure token
            </span>
            <span>
              <MessageSquareText size={18} /> Chat first
            </span>
          </div>
        </section>

        <section className="login-panel" aria-label="Đăng nhập">
          <div className="panel-heading">
            {authMode === "login" ? <LogIn size={20} /> : <UserPlus size={20} />}
            <span>{authMode === "login" ? "Student access" : "Create student account"}</span>
          </div>

          <div className="auth-switch" role="tablist" aria-label="Chọn chế độ xác thực">
            <button
              type="button"
              role="tab"
              aria-selected={authMode === "login"}
              className={authMode === "login" ? "active" : ""}
              onClick={() => switchAuthMode("login")}
            >
              <LogIn size={16} />
              Đăng nhập
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={authMode === "register"}
              className={authMode === "register" ? "active" : ""}
              onClick={() => switchAuthMode("register")}
            >
              <UserPlus size={16} />
              Đăng kí
            </button>
          </div>

          <form onSubmit={authMode === "login" ? handleLogin : handleRegister} className="login-form">
            <label>
              <span>MSSV</span>
              <div className="input-wrap">
                <UserRound size={18} />
                <input
                  value={mssv}
                  onChange={(event) => setMssv(event.target.value)}
                  placeholder="Nhập MSSV"
                  autoComplete="username"
                  required
                />
              </div>
            </label>
            {authMode === "register" && (
              <label>
                <span>Email</span>
                <div className="input-wrap">
                  <Mail size={18} />
                  <input
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    placeholder="Nhập email sinh viên"
                    type="email"
                    autoComplete="email"
                    required
                  />
                </div>
              </label>
            )}
            <label>
              <span>Mật khẩu</span>
              <div className="input-wrap">
                <LockKeyhole size={18} />
                <input
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Nhập mật khẩu"
                  type="password"
                  autoComplete={authMode === "login" ? "current-password" : "new-password"}
                  required
                />
              </div>
            </label>
            {authMode === "register" && (
              <label>
                <span>Xác nhận mật khẩu</span>
                <div className="input-wrap">
                  <ShieldCheck size={18} />
                  <input
                    value={confirmPassword}
                    onChange={(event) => setConfirmPassword(event.target.value)}
                    placeholder="Nhập lại mật khẩu"
                    type="password"
                    autoComplete="new-password"
                    required
                  />
                </div>
              </label>
            )}
            {registerMessage && <p className="success-message">{registerMessage}</p>}
            {loginError && <p className="error-message">{loginError}</p>}
            <button className="primary-button" type="submit" disabled={isLoggingIn}>
              {isLoggingIn
                ? authMode === "login"
                  ? "Đang kiểm tra..."
                  : "Đang tạo tài khoản..."
                : authMode === "login"
                  ? "Đăng nhập"
                  : "Đăng kí và vào hệ thống"}
            </button>
          </form>
        </section>
      </main>
    );
  }

  return (
    <main className="chat-shell">
      <aside className="sidebar" aria-label="Thông tin hệ thống">
        <div className="brand-row">
          <div className="mark small">
            <GraduationCap size={24} />
          </div>
          <div>
            <strong>Thesis Advising</strong>
            <span>AI consultation</span>
          </div>
        </div>
        <div className="status-block">
          <span>Trạng thái</span>
          <strong>Đã đăng nhập</strong>
        </div>

        <div className="upload-container">
          <h2 className="upload-title">Tải lên bảng điểm</h2>
          <p className="upload-subtitle">
            Cập nhật kết quả học tập để AI tư vấn khóa luận chính xác hơn (chấp nhận .csv, .xlsx)
          </p>

          <label className="file-select-label">
            <FileSpreadsheet size={18} />
            <span>Chọn file bảng điểm</span>
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={handleFileChange}
              className="file-select-input"
            />
          </label>

          {selectedFile && (
            <div className="selected-file-row">
              <Upload size={14} />
              <span className="selected-file-name">{selectedFile.name}</span>
            </div>
          )}

          {selectedFile && (
            <button
              className="upload-btn"
              type="button"
              disabled={isUploading}
              onClick={handleUploadTranscript}
            >
              {isUploading ? (
                <Loader2 size={16} className="spin" />
              ) : (
                <Upload size={16} />
              )}
              <span>{isUploading ? "Đang xử lý..." : "Xử lý bảng điểm"}</span>
            </button>
          )}

          {uploadSuccessMessage && (
            <div className="upload-result-box success">
              <p>{uploadSuccessMessage}</p>
              {(extractedGpa !== null || extractedPassedCount !== null) && (
                <div className="extracted-info">
                  {extractedGpa !== null && <div>GPA tích lũy: {extractedGpa}</div>}
                  {extractedPassedCount !== null && (
                    <div>Số môn đã đạt: {extractedPassedCount} môn</div>
                  )}
                </div>
              )}
            </div>
          )}

          {uploadErrorMessage && (
            <div className="upload-result-box error">
              <p>{uploadErrorMessage}</p>
            </div>
          )}
        </div>

        <button className="ghost-button" type="button" onClick={handleLogout}>
          <LogOut size={18} />
          Đăng xuất
        </button>
      </aside>

      <section className="chat-panel" aria-label="Hỏi đáp khóa luận">
        <header className="chat-header">
          <div>
            <p className="kicker">Academic copilot</p>
            <h1>Hỏi đáp khóa luận</h1>
          </div>
          <div className="live-pill">
            <span />
            Backend ready
          </div>
        </header>

        <div className="message-list" aria-live="polite">
          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <div className="message-avatar">
                {message.role === "user" ? <UserRound size={18} /> : <Sparkles size={18} />}
              </div>
              <div className="message-body">
                {message.isClarify ? (
                  <div style={{ 
                    backgroundColor: "#fff3cd", 
                    color: "#856404", 
                    padding: "12px", 
                    borderRadius: "8px", 
                    borderLeft: "4px solid #ffeeba", 
                    display: "flex", 
                    alignItems: "center", 
                    gap: "10px",
                    marginBottom: "12px"
                  }}>
                    <span style={{ fontSize: "1.2rem" }}>⚠️</span>
                    <strong>{message.content}</strong>
                  </div>
                ) : (
                  <p style={{ whiteSpace: "pre-wrap" }}>{message.content}</p>
                )}
                {message.sources && (
                  <div className="source-list">
                    {message.sources.map((source, index) => (
                      <div className="source-item" key={`${message.id}-${index}`}>
                        <span>Nguồn {index + 1}</span>
                        <p>{source}</p>
                      </div>
                    ))}
                  </div>
                )}

                {message.role === "assistant" && message.thinkingSteps && message.thinkingSteps.length > 0 && (
                  <div className="thinking-container">
                    <div
                      className={`thinking-header ${openThinkingMsgs[message.id] ? "open" : ""}`}
                      onClick={() => toggleThinking(message.id)}
                    >
                      <ChevronDown size={14} className="chevron" />
                      <span>Xem quá trình suy nghĩ</span>
                    </div>
                    {openThinkingMsgs[message.id] && (
                      <div className="thinking-body">
                        {message.thinkingSteps.map((step) => (
                          <div key={step.id} className={`thinking-step-item ${step.status}`}>
                            <div className="thinking-step-dot" />
                            <span>{step.label}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {message.role === "assistant" && message.debugTrace && message.debugTrace.length > 0 && (
                  <div className="trace-section">
                    <details className="trace-details">
                      <summary className="trace-summary">
                        <span>🛠️ Luồng xử lý Multi-Agent (Dành cho Giám khảo)</span>
                        <span>Mở rộng ▼</span>
                      </summary>
                      <div className="trace-content">
                        {message.criticScore !== undefined && (
                          <div className="critic-badge-row">
                            <span>🏅 Điểm đánh giá (Critic Score):</span>
                            <span className="score">{message.criticScore}/1.0</span>
                          </div>
                        )}
                        <div className="trace-steps-container">
                          {message.debugTrace.map((step, sIdx) => (
                            <div className="trace-step-card" key={`trace-${message.id}-${sIdx}`}>
                              <div className="trace-step-header">
                                <span className="trace-step-title">{step.task_id}</span>
                                <div className="trace-step-meta">
                                  <span className={`step-type-badge ${step.task_type.toLowerCase()}`}>
                                    {step.task_type}
                                  </span>
                                  <span className={`step-status-badge ${step.status.toLowerCase()}`}>
                                    {step.status}
                                  </span>
                                </div>
                              </div>
                              <div className="trace-step-body">
                                <div className="trace-step-query">
                                  Truy vấn: <strong>{step.query_intent}</strong>
                                </div>
                                <pre className="trace-step-data">{step.raw_data}</pre>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </details>
                  </div>
                )}
              </div>
            </article>
          ))}

          {isAsking && (
            <article className="message assistant">
              <div className="message-avatar">
                <Sparkles size={18} />
              </div>
              <div className="message-body loading-message-container">
                <div className="loading-message">
                  <span />
                  <p>AI đang chuẩn bị câu trả lời...</p>
                </div>
                
                {activeThinkingSteps.length > 0 && (
                  <div className="thinking-container">
                    <div className="thinking-header open">
                      <ChevronDown size={14} className="chevron" />
                      <span>Xem quá trình suy nghĩ</span>
                    </div>
                    <div className="thinking-body">
                      {activeThinkingSteps.map((step) => (
                        <div key={step.id} className={`thinking-step-item ${step.status}`}>
                          <div className="thinking-step-dot" />
                          <span>{step.label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </article>
          )}
        </div>

        {chatError && <p className="error-message chat-error">{chatError}</p>}

        <form className="composer" onSubmit={handleAsk}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Hỏi tôi về khóa luận..."
            rows={1}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <button type="submit" disabled={!canSubmitQuestion} aria-label="Gửi câu hỏi">
            <Send size={20} />
          </button>
        </form>
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
