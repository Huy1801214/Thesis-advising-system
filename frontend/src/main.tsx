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
  Trash2,
  Edit3,
  Sun,
  Moon,
  Menu,
  X,
  Copy,
  Check,
  FileText
} from "lucide-react";
import { login, register, sendQuestion, uploadTranscript, ChatTraceItem, sendQuestionStream, getSessions, getSessionMessages, getStudentProfile, deleteStudentProfile, updateStudentProfile, getCareersList } from "./api";
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

type Toast = {
  id: string;
  message: string;
  type: "success" | "error";
};

const TOKEN_STORAGE_KEY = "thesis_advising_token";
type AuthMode = "login" | "register";

function createId() {
  return crypto.randomUUID?.() || `msg_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function App() {
  const inputRef = React.useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = React.useRef<HTMLDivElement>(null);
  
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_STORAGE_KEY));
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [mssv, setMssv] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  
  // Custom toast list for beautiful notifications
  const [toasts, setToasts] = useState<Toast[]>([]);
  const showToast = (message: string, type: "success" | "error" = "success") => {
    const id = createId();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  };

  // Dark/Light Theme Switching
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem("theme");
    return saved === "dark" || (!saved && window.matchMedia("(prefers-color-scheme: dark)").matches);
  });

  // Sidebar collapsible state
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // History session renaming state
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingSessionTitle, setEditingSessionTitle] = useState("");

  const [question, setQuestion] = useState("");
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [activeThinkingSteps, setActiveThinkingSteps] = useState<ThinkingStep[]>([]);
  const [openThinkingMsgs, setOpenThinkingMsgs] = useState<{ [msgId: string]: boolean }>({});
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isUploadExpanded, setIsUploadExpanded] = useState(false);
  const [extractedGpa, setExtractedGpa] = useState<string | null>(null);
  const [extractedPassedCount, setExtractedPassedCount] = useState<number | null>(null);
  const [sessions, setSessions] = useState<{session_id: string, title: string}[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [chatMode, setChatMode] = useState<"agent" | "rag">("agent");
  const [isProfileExpanded, setIsProfileExpanded] = useState(false);
  const [studentMajor, setStudentMajor] = useState<string | null>(null);
  const [studentTargetCareer, setStudentTargetCareer] = useState<string | null>(null);
  const [studentInterests, setStudentInterests] = useState<string | null>(null);
  const [editMajor, setEditMajor] = useState("");
  const [editTargetCareer, setEditTargetCareer] = useState("");
  const [editInterests, setEditInterests] = useState("");
  const [careersList, setCareersList] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<"chat" | "profile">("chat");

  // Messages thread list
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

  // Sync dark theme class on document body
  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.body.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDarkMode]);

  // Auto-scroll chat thread to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeThinkingSteps, isAsking]);

  // Auto-growing Textarea input content logic
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 180)}px`;
    }
  }, [question]);

  const toggleThinking = (msgId: string) => {
    setOpenThinkingMsgs((prev) => ({
      ...prev,
      [msgId]: !prev[msgId],
    }));
  };

  const canSubmitQuestion = useMemo(
    () => question.trim().length > 0 && !isAsking,
    [isAsking, question],
  );

  const isNewChat = useMemo(
    () => messages.length === 0 || (messages.length === 1 && messages[0].id === "welcome"),
    [messages],
  );

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoggingIn(true);

    try {
      const result = await login(mssv.trim(), password);
      localStorage.setItem(TOKEN_STORAGE_KEY, result.access_token);
      setToken(result.access_token);
      setPassword("");
      showToast("Đăng nhập thành công!", "success");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "Đăng nhập không thành công.", "error");
    } finally {
      setIsLoggingIn(false);
    }
  }

  async function handleRegister(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (password !== confirmPassword) {
      showToast("Mật khẩu xác nhận không khớp.", "error");
      return;
    }

    setIsLoggingIn(true);

    try {
      await register(mssv.trim(), email.trim(), password);
      const result = await login(mssv.trim(), password);
      localStorage.setItem(TOKEN_STORAGE_KEY, result.access_token);
      setToken(result.access_token);
      setPassword("");
      setConfirmPassword("");
      showToast("Tạo tài khoản và đăng nhập thành công!", "success");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "Đăng kí không thành công.", "error");
    } finally {
      setIsLoggingIn(false);
    }
  }

  function switchAuthMode(nextMode: AuthMode) {
    setAuthMode(nextMode);
    setPassword("");
    setConfirmPassword("");
  }

  useEffect(() => {
    if (token) {
      loadSessions();
      loadStudentProfile();
      loadCareersList();
    }
  }, [token]);

  async function loadCareersList() {
    try {
      const res = await getCareersList(token!);
      if (res.status === "SUCCESS") {
        setCareersList(res.careers || []);
      }
    } catch (e) {
      console.error("Không thể tải danh sách ngành nghề tuyển dụng:", e);
    }
  }

  async function loadStudentProfile() {
    try {
      const res = await getStudentProfile(token!);
      if (res.status === "SUCCESS" && res.data) {
        setExtractedGpa(res.data.gpa ? res.data.gpa.toString() : null);
        setExtractedPassedCount(res.data.total_passed);
        setStudentMajor(res.data.major || null);
        setStudentTargetCareer(res.data.target_career || null);
        setStudentInterests(res.data.interests || null);
        setEditMajor(res.data.major || "");
        setEditTargetCareer(res.data.target_career || "");
        setEditInterests(res.data.interests || "");
      } else {
        setExtractedGpa(null);
        setExtractedPassedCount(null);
        setStudentMajor(null);
        setStudentTargetCareer(null);
        setStudentInterests(null);
        setEditMajor("");
        setEditTargetCareer("");
        setEditInterests("");
      }
    } catch (e) {
      console.error("Không thể tải bảng điểm đã lưu:", e);
    }
  }

  async function handleUpdateProfile(major: string, targetCareer: string, interests: string) {
    if (!token) return;
    try {
      const res = await updateStudentProfile(token, major, targetCareer, interests);
      if (res.status === "SUCCESS") {
        showToast("Cập nhật hồ sơ cá nhân thành công!", "success");
        setStudentMajor(major || null);
        setStudentTargetCareer(targetCareer || null);
        setStudentInterests(interests || null);
        setIsProfileExpanded(false);
      }
    } catch (error) {
      showToast(error instanceof Error ? error.message : "Cập nhật hồ sơ thất bại.", "error");
    }
  }

  async function loadSessions() {
    try {
      const data = await getSessions(token!);
      setSessions(data.sessions || []);
    } catch (e) {
      console.error(e);
    }
  }

  async function handleSelectSession(sessionId: string) {
    setCurrentSessionId(sessionId);
    setViewMode("chat");
    try {
      const data = await getSessionMessages(sessionId, token!);
      const loadedMessages: ChatMessage[] = data.messages.map((m: any) => ({
        id: m.id.toString(),
        role: m.role,
        content: m.content,
      }));
      setMessages(loadedMessages);
      
      // Auto-collapse sidebar on mobile screen size after selection
      if (window.innerWidth <= 768) {
        setIsSidebarOpen(false);
      }
    } catch (e) {
      showToast("Không thể tải tin nhắn của đoạn chat này.", "error");
    }
  }

  // Local Session Management (Delete & Rename)
  const handleDeleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
    showToast("Đã xóa đoạn hội thoại.", "success");
    if (currentSessionId === sessionId) {
      handleNewChat();
    }
  };

  const handleStartRename = (sessionId: string, title: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingSessionId(sessionId);
    setEditingSessionTitle(title);
  };

  const handleSaveRename = (sessionId: string) => {
    if (editingSessionTitle.trim()) {
      setSessions((prev) =>
        prev.map((s) => (s.session_id === sessionId ? { ...s, title: editingSessionTitle.trim() } : s))
      );
      showToast("Đã đổi tên cuộc trò chuyện thành công.", "success");
    }
    setEditingSessionId(null);
  };

  function handleNewChat() {
    setCurrentSessionId(null);
    setViewMode("chat");
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: "Chào mừng bạn đến với hệ thống tư vấn khóa luận.",
        sources: ["Hãy đặt câu hỏi về đề tài, quy trình, điều kiện thực hiện..."],
      },
    ]);
    if (window.innerWidth <= 768) {
      setIsSidebarOpen(false);
    }
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
    setIsAsking(true);
    setActiveThinkingSteps([]);

    let finalAns = "";
    let criticScr = 0;
    let debugTrc: ChatTraceItem[] = [];

    try {
      await sendQuestionStream(prompt, token, currentSessionId, chatMode, (event, data) => {
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

          if (data.session_id) {
            if (currentSessionId !== data.session_id) {
              setCurrentSessionId(data.session_id);
              loadSessions(); 
            }
          }
        }
        else if (event === "error") {
          showToast(`Lỗi từ máy chủ: ${data.message}`, "error");
        }
      });

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
      showToast(message, "error");

      if (message.toLowerCase().includes("đăng nhập")) {
        handleLogout();
      }
    } finally {
      setIsAsking(false);
      setActiveThinkingSteps([]);
    }
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      showToast(`Đã chọn file: ${file.name}`, "success");
    }
  }

  async function handleUploadTranscript() {
    if (!selectedFile || !token) return;
    setIsUploading(true);

    try {
      const result = await uploadTranscript(selectedFile, token);
      showToast(result.message || "Đồng bộ bảng điểm thành công!", "success");
      if (result.data) {
        setExtractedGpa(result.data.gpa);
        setExtractedPassedCount(result.data.total_passed);
      }
      setSelectedFile(null);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "Tải lên bảng điểm thất bại.", "error");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleDeleteTranscript() {
    if (!token) return;
    if (!confirm("Bạn có chắc chắn muốn xóa dữ liệu bảng điểm khỏi hệ thống không? Hành động này sẽ không thể hoàn tác.")) {
      return;
    }
    
    try {
      const res = await deleteStudentProfile(token);
      if (res.status === "SUCCESS") {
        setExtractedGpa(null);
        setExtractedPassedCount(null);
        setSelectedFile(null);
        showToast("Đã xóa dữ liệu bảng điểm thành công!", "success");
      }
    } catch (error) {
      showToast(error instanceof Error ? error.message : "Xóa bảng điểm thất bại.", "error");
    }
  }

  function handleLogout() {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    setToken(null);
    setSelectedFile(null);
    setExtractedGpa(null);
    setExtractedPassedCount(null);
    setActiveThinkingSteps([]);
    setOpenThinkingMsgs({});
    setSessions([]);
    setCurrentSessionId(null);
    showToast("Đã đăng xuất khỏi hệ thống.", "success");
  }

  // Custom code formatter block to highlight backticks in monospace
  function formatMessageText(text: string) {
    if (!text) return "";
    
    // Helper to parse double asterisks **bold** inside a text block
    function parseBoldAndText(str: string, subIdx: number) {
      if (!str) return "";
      const boldParts = str.split(/(\*\*[\s\S]*?\*\*)/g);
      return (
        <span key={subIdx}>
          {boldParts.map((subPart, idx) => {
            if (subPart.startsWith("**") && subPart.endsWith("**")) {
              const cleanText = subPart.slice(2, -2);
              return (
                <strong 
                  key={idx} 
                  className="highlight-bold" 
                  style={{ 
                    color: "var(--primary)", 
                    fontWeight: 700 
                  }}
                >
                  {cleanText}
                </strong>
              );
            }
            return subPart;
          })}
        </span>
      );
    }

    const parts = text.split(/(```[\s\S]*?```)/g);
    return parts.map((part, index) => {
      if (part.startsWith("```") && part.endsWith("```")) {
        const rawCode = part.slice(3, -3);
        const lines = rawCode.trim().split("\n");
        let language = "";
        let codeContent = lines;
        
        // Detect language name
        if (lines[0] && !lines[0].includes(" ") && lines[0].length < 15) {
          language = lines[0];
          codeContent = lines.slice(1);
        }
        
        const codeText = codeContent.join("\n");
        return (
          <pre key={index}>
            <div className="code-block-header" style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "6px 12px",
              background: "#1e293b",
              color: "#94a3b8",
              fontSize: "0.75rem",
              borderTopLeftRadius: "6px",
              borderTopRightRadius: "6px",
              borderBottom: "1px solid #334155",
              fontFamily: "sans-serif"
            }}>
              <span>{language.toUpperCase() || "CODE"}</span>
              <button
                type="button"
                onClick={() => {
                  navigator.clipboard.writeText(codeText);
                  showToast("Đã sao chép mã nguồn!", "success");
                }}
                style={{ fontSize: "0.7rem", color: "#f1f5f9", display: "flex", alignItems: "center", gap: "4px" }}
              >
                <Copy size={12} /> Sao chép
              </button>
            </div>
            <code>{codeText}</code>
          </pre>
        );
      }
      
      // Inline code blocks format (single backtick)
      const inlineParts = part.split(/(`[^`]+`)/g);
      return (
        <span key={index}>
          {inlineParts.map((subPart, subIdx) => {
            if (subPart.startsWith("`") && subPart.endsWith("`")) {
              return <code key={subIdx}>{subPart.slice(1, -1)}</code>;
            }
            return parseBoldAndText(subPart, subIdx);
          })}
        </span>
      );
    });
  }

  // -------------------------------------------------------------
  // RENDER: Login State
  // -------------------------------------------------------------
  if (!token) {
    return (
      <main className="login-shell">
        {/* Toast Display Overlay */}
        <div className="toast-container">
          {toasts.map((t) => (
            <div key={t.id} className={`toast ${t.type}`}>
              <span>{t.type === "success" ? "✓" : "⚠️"}</span>
              <p>{t.message}</p>
            </div>
          ))}
        </div>

        <section className="login-panel" aria-label="Đăng nhập">
          <div className="brand-logo-container" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px", marginBottom: "2rem" }}>
            <div className="mark" style={{ width: "48px", height: "48px", borderRadius: "12px", background: "var(--primary-light)", color: "var(--primary)", display: "grid", placeItems: "center", boxShadow: "var(--shadow-sm)" }}>
              <GraduationCap size={24} />
            </div>
            <span className="brand-name" style={{ fontSize: "1.35rem", fontWeight: 800, color: "var(--text-main)", letterSpacing: "-0.02em" }}>Thesis Advising AI</span>
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 550 }}>Hệ thống tư vấn khóa luận tốt nghiệp</span>
          </div>

          <div className="auth-switch" role="tablist" aria-label="Chọn chế độ xác thực" style={{ marginBottom: "1.5rem" }}>
            <button
              type="button"
              role="tab"
              aria-selected={authMode === "login"}
              className={authMode === "login" ? "active" : ""}
              onClick={() => switchAuthMode("login")}
            >
              <LogIn size={14} /> Đăng nhập
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={authMode === "register"}
              className={authMode === "register" ? "active" : ""}
              onClick={() => switchAuthMode("register")}
            >
              <UserPlus size={14} /> Đăng ký
            </button>
          </div>

          <form onSubmit={authMode === "login" ? handleLogin : handleRegister} className="login-form">
            <label>
              <span>MÃ SỐ SINH VIÊN (MSSV)</span>
              <div className="input-wrap">
                <UserRound size={18} />
                <input
                  value={mssv}
                  onChange={(event) => setMssv(event.target.value)}
                  placeholder="Nhập MSSV (ví dụ: 1801214)"
                  autoComplete="username"
                  required
                />
              </div>
            </label>

            {authMode === "register" && (
              <label>
                <span>EMAIL SINH VIÊN</span>
                <div className="input-wrap">
                  <Mail size={18} />
                  <input
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    placeholder="email@student.edu.vn"
                    type="email"
                    autoComplete="email"
                    required
                  />
                </div>
              </label>
            )}

            <label>
              <span>MẬT KHẨU</span>
              <div className="input-wrap">
                <LockKeyhole size={18} />
                <input
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Nhập mật khẩu truy cập"
                  type="password"
                  autoComplete={authMode === "login" ? "current-password" : "new-password"}
                  required
                />
              </div>
            </label>

            {authMode === "register" && (
              <label>
                <span>XÁC NHẬN MẬT KHẨU</span>
                <div className="input-wrap">
                  <ShieldCheck size={18} />
                  <input
                    value={confirmPassword}
                    onChange={(event) => setConfirmPassword(event.target.value)}
                    placeholder="Nhập lại mật khẩu để kiểm tra"
                    type="password"
                    autoComplete="new-password"
                    required
                  />
                </div>
              </label>
            )}

            <button className="primary-button" type="submit" disabled={isLoggingIn} style={{ marginTop: "0.5rem" }}>
              {isLoggingIn
                ? authMode === "login"
                  ? "Đang xác thực..."
                  : "Đang đăng ký..."
                : authMode === "login"
                  ? "Đăng nhập"
                  : "Đăng ký thành viên"}
            </button>
          </form>

          <div style={{ textAlign: "center", fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2rem" }}>
            © 2026 Thesis Advising System. Powered by Graph RAG.
          </div>
        </section>
      </main>
    );
  }

  // -------------------------------------------------------------
  // RENDER: Chat Workspace State
  // -------------------------------------------------------------
  return (
    <main className="chat-shell">
      {/* Toast Display Overlay */}
      <div className="toast-container">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.type}`}>
            <span>{t.type === "success" ? "✓" : "⚠️"}</span>
            <p>{t.message}</p>
          </div>
        ))}
      </div>

      {/* Sidebar Overlay Backdrop on Mobile */}
      <div 
        className={`sidebar-overlay ${isSidebarOpen ? "active" : ""}`} 
        onClick={() => setIsSidebarOpen(false)} 
      />

      <aside className={`sidebar ${isSidebarOpen ? "" : "collapsed"}`} aria-label="Thông tin hệ thống">
        <div className="brand-row">
          <div className="mark small">
            <GraduationCap size={20} />
          </div>
          <div className="brand-info">
            <strong>Thesis Advising</strong>
            <span>Academic Copilot</span>
          </div>
          <button 
            type="button" 
            className="sidebar-close-btn" 
            onClick={() => setIsSidebarOpen(false)}
            title="Đóng sidebar"
          >
            <Menu size={16} />
          </button>
        </div>

        <div className="history-container">
          <button 
            type="button" 
            className="new-chat-btn" 
            onClick={handleNewChat}
          >
            <MessageSquareText size={16} />
            Hội thoại mới
          </button>
          
          <div className="session-list">
            <span className="session-section-title">Lịch sử hội thoại</span>
            {sessions.map(s => (
              <div 
                key={s.session_id} 
                className={`session-item-wrapper ${currentSessionId === s.session_id ? "active" : ""}`}
              >
                {editingSessionId === s.session_id ? (
                  <input
                    type="text"
                    className="session-rename-input"
                    value={editingSessionTitle}
                    onChange={(e) => setEditingSessionTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleSaveRename(s.session_id);
                      if (e.key === "Escape") setEditingSessionId(null);
                    }}
                    onBlur={() => handleSaveRename(s.session_id)}
                    autoFocus
                  />
                ) : (
                  <>
                    <button 
                      onClick={() => handleSelectSession(s.session_id)}
                      className="session-btn"
                      title={s.title}
                    >
                      <MessageSquareText size={14} style={{ flexShrink: 0 }} />
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.title}</span>
                    </button>
                    <div className="session-actions">
                      <button 
                        type="button"
                        className="session-action-btn"
                        onClick={(e) => handleStartRename(s.session_id, s.title, e)}
                        title="Đổi tên"
                      >
                        <Edit3 size={12} />
                      </button>
                      <button 
                        type="button"
                        className="session-action-btn delete"
                        onClick={(e) => handleDeleteSession(s.session_id, e)}
                        title="Xóa"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Collapsible Upload Transcript Panel */}
        <div className="sidebar-accordion">
          <button 
            type="button" 
            className={`accordion-header ${isUploadExpanded ? "active" : ""}`}
            onClick={() => setIsUploadExpanded(!isUploadExpanded)}
            title="Đồng bộ bảng điểm & GPA"
          >
            <FileSpreadsheet size={16} style={{ flexShrink: 0 }} />
            <span>Bảng điểm & GPA</span>
            <ChevronDown 
              size={14} 
              style={{ 
                marginLeft: "auto", 
                transform: isUploadExpanded ? "rotate(180deg)" : "none", 
                transition: "transform 0.2s" 
              }} 
            />
          </button>

          {(extractedGpa !== null || extractedPassedCount !== null) && !isUploadExpanded && (
            <div className="extracted-summary-row">
              {extractedGpa !== null && <span className="stat-pill">GPA: {extractedGpa}</span>}
              {extractedPassedCount !== null && <span className="stat-pill">{extractedPassedCount} TC</span>}
            </div>
          )}

          {isUploadExpanded && (
            <div className="accordion-content">
              <p className="upload-subtitle">
                Tải lên bảng điểm để AI tự trích xuất GPA và số tín chỉ đạt (chấp nhận .csv, .xlsx, .xls, .pdf, .docx, .doc).
              </p>

              <label className="file-select-label">
                <Upload size={14} />
                <span>{selectedFile ? "Đổi tệp" : "Chọn tệp"}</span>
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls,.pdf,.docx,.doc"
                  onChange={handleFileChange}
                  className="file-select-input"
                />
              </label>

              {selectedFile && (
                <div className="selected-file-row">
                  <FileText size={12} style={{ color: "var(--primary)" }} />
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
                    <Loader2 size={14} className="spin" />
                  ) : (
                    <Upload size={14} />
                  )}
                  <span>{isUploading ? "Đang xử lý..." : "Xử lý bảng điểm"}</span>
                </button>
              )}

              {(extractedGpa !== null || extractedPassedCount !== null) && (
                <div style={{ marginTop: "4px", fontSize: "0.75rem", borderTop: "1px solid var(--border-color)", paddingTop: "8px", display: "flex", flexDirection: "column", gap: "8px" }}>
                  <div>
                    <div style={{ fontWeight: 700, color: "var(--text-main)", marginBottom: "4px" }}>Đã đồng bộ:</div>
                    {extractedGpa !== null && <div>• GPA tích lũy: <strong style={{ color: "var(--primary)" }}>{extractedGpa}</strong></div>}
                    {extractedPassedCount !== null && <div>• Tín chỉ đạt: <strong style={{ color: "var(--primary)" }}>{extractedPassedCount}</strong></div>}
                  </div>
                  <button
                    className="delete-transcript-btn"
                    type="button"
                    onClick={handleDeleteTranscript}
                    style={{
                      width: "100%",
                      height: "28px",
                      backgroundColor: "var(--danger-light)",
                      color: "var(--danger)",
                      borderRadius: "6px",
                      fontSize: "0.72rem",
                      fontWeight: 700,
                      gap: "4px",
                      marginTop: "4px"
                    }}
                  >
                    Xóa dữ liệu bảng điểm
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Collapsible Profile & Goals Panel */}
        <div className="sidebar-accordion" style={{ borderTop: "1px solid var(--border-color)", paddingTop: "8px", marginTop: "8px" }}>
          <button 
            type="button" 
            className={`accordion-header ${isProfileExpanded ? "active" : ""}`}
            onClick={() => setIsProfileExpanded(!isProfileExpanded)}
            title="Định hướng nghề nghiệp & Chuyên ngành"
          >
            <GraduationCap size={16} style={{ flexShrink: 0 }} />
            <span>Hồ sơ & Định hướng</span>
            <ChevronDown 
              size={14} 
              style={{ 
                marginLeft: "auto", 
                transform: isProfileExpanded ? "rotate(180deg)" : "none", 
                transition: "transform 0.2s" 
              }} 
            />
          </button>

          {!isProfileExpanded && (studentMajor || studentTargetCareer) && (
            <div className="extracted-summary-row" style={{ marginTop: "4px" }}>
              {studentMajor && <span className="stat-pill" style={{ fontSize: "0.68rem" }}>{studentMajor}</span>}
              {studentTargetCareer && <span className="stat-pill" style={{ fontSize: "0.68rem" }} title={studentTargetCareer}>{studentTargetCareer.split(" (")[0]}</span>}
            </div>
          )}

          {isProfileExpanded && (
            <div className="accordion-content" style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "8px" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <label style={{ fontSize: "0.68rem", fontWeight: 700, color: "var(--text-muted)", display: "block" }}>Chuyên ngành:</label>
                <select
                  value={editMajor}
                  onChange={(e) => setEditMajor(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "6px 8px",
                    borderRadius: "6px",
                    border: "1px solid var(--border-color)",
                    backgroundColor: "var(--bg-card)",
                    color: "var(--text-main)",
                    fontSize: "0.72rem",
                    outline: "none"
                  }}
                >
                  <option value="">-- Chọn chuyên ngành --</option>
                  <option value="Kỹ thuật phần mềm">Kỹ thuật phần mềm</option>
                  <option value="Hệ thống thông tin">Hệ thống thông tin</option>
                  <option value="Công nghệ thông tin">Công nghệ thông tin</option>
                  <option value="Mạng máy tính và Truyền thông">Mạng máy tính & Truyền thông</option>
                </select>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <label style={{ fontSize: "0.68rem", fontWeight: 700, color: "var(--text-muted)", display: "block" }}>Định hướng nghề nghiệp:</label>
                <select
                  value={editTargetCareer}
                  onChange={(e) => setEditTargetCareer(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "6px 8px",
                    borderRadius: "6px",
                    border: "1px solid var(--border-color)",
                    backgroundColor: "var(--bg-card)",
                    color: "var(--text-main)",
                    fontSize: "0.72rem",
                    outline: "none"
                  }}
                >
                  <option value="">-- Chọn mục tiêu nghề nghiệp --</option>
                  {careersList.map(career => (
                    <option key={career} value={career}>{career}</option>
                  ))}
                </select>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <label style={{ fontSize: "0.68rem", fontWeight: 700, color: "var(--text-muted)", display: "block" }}>Sở thích công nghệ:</label>
                <input
                  type="text"
                  value={editInterests}
                  onChange={(e) => setEditInterests(e.target.value)}
                  placeholder="Ví dụ: Kubernetes, Cloud, AI..."
                  style={{
                    width: "100%",
                    padding: "6px 8px",
                    borderRadius: "6px",
                    border: "1px solid var(--border-color)",
                    backgroundColor: "var(--bg-card)",
                    color: "var(--text-main)",
                    fontSize: "0.72rem",
                    outline: "none"
                  }}
                />
              </div>

              <button
                className="upload-btn"
                type="button"
                onClick={() => handleUpdateProfile(editMajor, editTargetCareer, editInterests)}
                style={{ width: "100%", height: "28px", display: "flex", alignItems: "center", justifyContent: "center", gap: "4px" }}
              >
                <span>Lưu thông tin</span>
              </button>
            </div>
          )}
        </div>

        <div className="logout-btn-container" style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <button 
            className="logout-btn" 
            type="button" 
            onClick={() => setViewMode(viewMode === "profile" ? "chat" : "profile")}
            style={{
              backgroundColor: viewMode === "profile" ? "var(--primary-light)" : "transparent",
              color: viewMode === "profile" ? "var(--primary)" : "var(--text-main)",
              borderColor: viewMode === "profile" ? "var(--primary)" : "transparent"
            }}
          >
            <UserRound size={16} />
            <span>Hồ sơ & Định hướng</span>
          </button>
          <button className="logout-btn" type="button" onClick={handleLogout}>
            <LogOut size={16} />
            <span>Đăng xuất</span>
          </button>
        </div>
      </aside>

      {viewMode === "profile" ? (
        <section className="chat-panel" aria-label="Hồ sơ cá nhân">
          <header className="chat-header">
            <div className="chat-header-left">
              <div className="chat-header-title">
                <p className="kicker">Thiết lập tài khoản</p>
                <h1>Hồ sơ cá nhân & Định hướng</h1>
              </div>
            </div>
            <div className="chat-header-right">
              <button 
                type="button" 
                onClick={() => setViewMode("chat")}
                style={{
                  fontSize: "0.78rem",
                  padding: "8px 16px",
                  borderRadius: "8px",
                  backgroundColor: "var(--primary)",
                  color: "#fff",
                  fontWeight: 600,
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  cursor: "pointer",
                  border: "none"
                }}
              >
                <span>Quay lại phòng chat</span>
              </button>
            </div>
          </header>

          <div className="profile-container-scrollable" style={{ padding: "30px", overflowY: "auto", flex: 1, backgroundColor: "var(--bg-app)" }}>
            <div className="profile-card-layout" style={{ maxWidth: "800px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "24px" }}>
              
              {/* Thống kê học vụ */}
              <div className="stat-cards-wrapper" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "16px" }}>
                <div className="stat-card-item" style={{ padding: "20px", borderRadius: "12px", backgroundColor: "var(--bg-card)", border: "1px solid var(--border-color)", boxShadow: "var(--shadow-sm)" }}>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px" }}>Điểm trung bình (GPA)</span>
                  <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--primary)", marginTop: "8px" }}>
                    {extractedGpa || "Chưa có"}
                  </div>
                  <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "6px" }}>Được tự động bóc tách từ file bảng điểm tích lũy của bạn.</p>
                </div>
                
                <div className="stat-card-item" style={{ padding: "20px", borderRadius: "12px", backgroundColor: "var(--bg-card)", border: "1px solid var(--border-color)", boxShadow: "var(--shadow-sm)" }}>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px" }}>Tín chỉ tích lũy</span>
                  <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--success)", marginTop: "8px" }}>
                    {extractedPassedCount || 0} <span style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--text-muted)" }}>/ 130 tín chỉ</span>
                  </div>
                  <div style={{ width: "100%", height: "6px", backgroundColor: "var(--border-color)", borderRadius: "3px", marginTop: "12px", overflow: "hidden" }}>
                    <div style={{ width: `${Math.min(((extractedPassedCount || 0) / 130) * 100, 100)}%`, height: "100%", backgroundColor: "var(--success)" }} />
                  </div>
                </div>
              </div>

              {/* Form chỉnh sửa */}
              <div className="profile-form-card" style={{ padding: "30px", borderRadius: "12px", backgroundColor: "var(--bg-card)", border: "1px solid var(--border-color)", boxShadow: "var(--shadow-md)", display: "flex", flexDirection: "column", gap: "20px" }}>
                <h3 style={{ fontSize: "1.1rem", fontWeight: 700, borderBottom: "1px solid var(--border-color)", paddingBottom: "10px", color: "var(--text-main)", margin: 0 }}>Định hướng học tập & Nghề nghiệp</h3>
                
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    <label style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)" }}>Mã số sinh viên (MSSV):</label>
                    <input 
                      type="text" 
                      value={mssv} 
                      disabled 
                      style={{ padding: "10px 12px", borderRadius: "8px", border: "1px solid var(--border-color)", backgroundColor: "var(--bg-app)", color: "var(--text-muted)", fontSize: "0.85rem", cursor: "not-allowed", outline: "none" }}
                    />
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    <label style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)" }}>Email liên hệ:</label>
                    <input 
                      type="text" 
                      value={email || `${mssv}@hcmuaf.edu.vn`} 
                      disabled 
                      style={{ padding: "10px 12px", borderRadius: "8px", border: "1px solid var(--border-color)", backgroundColor: "var(--bg-app)", color: "var(--text-muted)", fontSize: "0.85rem", cursor: "not-allowed", outline: "none" }}
                    />
                  </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <label style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)" }}>Chuyên ngành tuyển sinh:</label>
                  <select
                    value={editMajor}
                    onChange={(e) => setEditMajor(e.target.value)}
                    style={{ padding: "10px 12px", borderRadius: "8px", border: "1px solid var(--border-color)", backgroundColor: "var(--bg-card)", color: "var(--text-main)", fontSize: "0.85rem", outline: "none" }}
                  >
                    <option value="">-- Chọn chuyên ngành học --</option>
                    <option value="Kỹ thuật phần mềm">Kỹ thuật phần mềm</option>
                    <option value="Hệ thống thông tin">Hệ thống thông tin</option>
                    <option value="Công nghệ thông tin">Công nghệ thông tin</option>
                    <option value="Mạng máy tính và Truyền thông">Mạng máy tính & Truyền thông</option>
                  </select>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <label style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)" }}>Mục tiêu công việc mong muốn (Lấy động từ CSDL):</label>
                  <select
                    value={editTargetCareer}
                    onChange={(e) => setEditTargetCareer(e.target.value)}
                    style={{ padding: "10px 12px", borderRadius: "8px", border: "1px solid var(--border-color)", backgroundColor: "var(--bg-card)", color: "var(--text-main)", fontSize: "0.85rem", outline: "none" }}
                  >
                    <option value="">-- Chọn mục tiêu nghề nghiệp --</option>
                    {careersList.map(c => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <label style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)" }}>Sở thích công nghệ / Từ khóa cá nhân:</label>
                  <input
                    type="text"
                    value={editInterests}
                    onChange={(e) => setEditInterests(e.target.value)}
                    placeholder="Ví dụ: Kubernetes, Cloud Computing, Trí tuệ nhân tạo, ReactJS, CI/CD..."
                    style={{ padding: "10px 12px", borderRadius: "8px", border: "1px solid var(--border-color)", backgroundColor: "var(--bg-card)", color: "var(--text-main)", fontSize: "0.85rem", outline: "none" }}
                  />
                </div>

                <button
                  type="button"
                  onClick={() => handleUpdateProfile(editMajor, editTargetCareer, editInterests)}
                  style={{
                    backgroundColor: "var(--primary)",
                    color: "#fff",
                    border: "none",
                    padding: "12px",
                    borderRadius: "8px",
                    fontWeight: 700,
                    cursor: "pointer",
                    fontSize: "0.9rem",
                    transition: "background-color 0.2s"
                  }}
                  onMouseOver={(e) => e.currentTarget.style.backgroundColor = "var(--primary-hover)"}
                  onMouseOut={(e) => e.currentTarget.style.backgroundColor = "var(--primary)"}
                >
                  Lưu thông tin hồ sơ
                </button>
              </div>

            </div>
          </div>
        </section>
      ) : (
        <section className="chat-panel" aria-label="Hỏi đáp khóa luận">
          <header className="chat-header">
            <div className="chat-header-left">
              {!isSidebarOpen && (
                <button 
                  className="sidebar-toggle-btn" 
                  onClick={() => setIsSidebarOpen(true)}
                  title="Mở sidebar"
                >
                  <Menu size={18} />
                </button>
              )}
              <div className="chat-header-title">
                <p className="kicker">Hỗ trợ học tập</p>
                <h1>Tư vấn khóa luận tốt nghiệp</h1>
              </div>
            </div>
            
            <div className="chat-header-right">
              <div className="model-selector">
                <button 
                  type="button" 
                  className={`selector-btn ${chatMode === "agent" ? "active" : ""}`}
                  onClick={() => setChatMode("agent")}
                  title="Chế độ Multi-Agent"
                >
                  <Sparkles size={14} />
                  <span>Multi-Agent</span>
                </button>
                <button 
                  type="button" 
                  className={`selector-btn ${chatMode === "rag" ? "active" : ""}`}
                  onClick={() => setChatMode("rag")}
                  title="Chế độ Thuần RAG (Baseline)"
                >
                  <BookOpenCheck size={14} />
                  <span>Thuần RAG</span>
                </button>
              </div>

              <div className="live-pill">
                <span />
                AI Core Online
              </div>
              
              <button 
                className="theme-toggle-btn" 
                onClick={() => setIsDarkMode(!isDarkMode)}
                title={isDarkMode ? "Chuyển sang Giao diện Sáng" : "Chuyển sang Giao diện Tối"}
              >
                {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
              </button>
            </div>
          </header>

          {isNewChat ? (
            <div className="new-chat-workspace">
              <div className="new-chat-center">
                <h2 className="new-chat-title">Chúng ta nên bắt đầu từ đâu?</h2>
                <form className="composer centered-composer" onSubmit={handleAsk}>
                  <textarea
                    ref={inputRef}
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Hỏi tôi bất kỳ điều gì về khóa luận..."
                    rows={1}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        event.currentTarget.form?.requestSubmit();
                      }
                    }}
                  />
                  <button 
                    type="submit" 
                    className="send-btn"
                    disabled={!canSubmitQuestion} 
                    aria-label="Gửi câu hỏi"
                  >
                    <Send size={16} />
                  </button>
                </form>
                <div className="suggestion-prompts">
                  <div 
                    onClick={() => setQuestion("Điều kiện đăng ký làm khóa luận tốt nghiệp là gì?")} 
                    className="suggestion-card"
                  >
                    <span>Đăng ký khóa luận</span>
                    <p>Tìm hiểu điều kiện và GPA tối thiểu...</p>
                  </div>
                  <div 
                    onClick={() => setQuestion("Quy trình thực hiện khóa luận tốt nghiệp gồm những bước nào?")} 
                    className="suggestion-card"
                  >
                    <span>Quy trình thực hiện</span>
                    <p>Các mốc thời gian và hồ sơ cần chuẩn bị...</p>
                  </div>
                  <div 
                    onClick={() => setQuestion("Làm sao để đồng bộ bảng điểm tích lũy vào hệ thống tư vấn?")} 
                    className="suggestion-card"
                  >
                    <span>Đồng bộ bảng điểm</span>
                    <p>Hướng dẫn tải lên tệp CSV/Excel...</p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <>
              <div className="message-list">
                {messages.map((message) => (
                  <article key={message.id} className={`message ${message.role}`}>
                    <div className="message-avatar">
                      {message.role === "user" ? <UserRound size={16} /> : <Sparkles size={16} />}
                    </div>
                    <div className="message-body">
                      <div className="message-bubble">
                        {message.isClarify ? (
                          <div className="clarify-box">
                            <span className="clarify-icon">⚠️</span>
                            <strong style={{ fontSize: "0.9rem" }}>{message.content}</strong>
                          </div>
                        ) : (
                          <div>{formatMessageText(message.content)}</div>
                        )}
                      </div>

                      {message.sources && message.sources.length > 0 && (
                        <div className="source-list">
                          {message.sources.map((source, index) => (
                            <div className="source-item" key={`${message.id}-${index}`}>
                              <span>Tài liệu tham khảo {index + 1}</span>
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
                            <ChevronDown size={12} className="chevron" />
                            <span>Xem quá trình xử lý Agent</span>
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
                                  <span>Điểm phản biện (Critic Score):</span>
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
                                        Nội dung: <strong>{step.query_intent}</strong>
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
                      <Sparkles size={16} />
                    </div>
                    <div className="message-body">
                      <div className="message-bubble" style={{ display: "inline-block" }}>
                        <div className="typing-indicator">
                          <span />
                          <span />
                          <span />
                        </div>
                      </div>
                      
                      {activeThinkingSteps.length > 0 && (
                        <div className="thinking-container">
                          <div className="thinking-header open">
                            <ChevronDown size={12} className="chevron" />
                            <span>Xem quá trình xử lý Agent</span>
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
                
                <div ref={messagesEndRef} />
              </div>

              <div className="composer-container">
                <form className="composer" onSubmit={handleAsk}>
                  <textarea
                    ref={inputRef}
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Nhập câu hỏi của bạn về khóa luận, quy chế học tập..."
                    rows={1}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        event.currentTarget.form?.requestSubmit();
                      }
                    }}
                  />
                  <button 
                    type="submit" 
                    className="send-btn"
                    disabled={!canSubmitQuestion} 
                    aria-label="Gửi câu hỏi"
                  >
                    <Send size={16} />
                  </button>
                </form>
                <div style={{ textAlign: "center", fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "4px" }}>
                  Hệ thống RAG và Graph RAG có thể cung cấp thông tin sai lệch. Hãy kiểm chứng lại nếu cần.
                </div>
              </div>
            </>
          )}
        </section>
      )}
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
