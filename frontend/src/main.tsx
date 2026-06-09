import React, { FormEvent, useMemo, useState } from "react";
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
} from "lucide-react";
import { login, register, sendQuestion } from "./api";
import "./styles.css";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
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

    try {
      const response = await sendQuestion(prompt, token);
      const finalAnswer =
        response.answer ||
        (response.data.length > 0 ? response.data.map(formatResult).join("\n\n") : "");
      const traceNotes = response.trace?.map(
        (item) =>
          `${item.task_id} | ${item.worker} | ${item.status}\n${item.sub_question}\n${item.answer}`,
      );

      setMessages((current) => [
        ...current,
        {
          id: response.session_id,
          role: "assistant",
          content: finalAnswer || "Chưa tìm thấy kết quả phù hợp.",
          sources: traceNotes && traceNotes.length > 1 ? traceNotes : undefined,
        },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Không thể gửi câu hỏi.";
      setChatError(message);

      if (message.toLowerCase().includes("đăng nhập")) {
        handleLogout();
      }
    } finally {
      setIsAsking(false);
    }
  }

  function handleLogout() {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    setToken(null);
    setChatError("");
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
                <p>{message.content}</p>
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
              </div>
            </article>
          ))}

          {isAsking && (
            <article className="message assistant">
              <div className="message-avatar">
                <Sparkles size={18} />
              </div>
              <div className="message-body loading-message">
                <span />
                <p>AI đang suy nghĩ và truy xuất dữ liệu...</p>
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
