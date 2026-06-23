const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

export type LoginResponse = {
  access_token: string;
  token_type: string;
};

export type RegisterResponse = {
  mssv: string;
  email: string;
  is_active: boolean;
};

export type ChatTraceItem = {
  task_id: string;
  task_type: "RAG" | "GRAG";
  query_intent: string;
  raw_data: string;
  status: "SUCCESS" | "FAILED";
  error_log?: string;
};

export type ChatResponse = {
  answer: string;
  critic_score: number;
  debug_trace: ChatTraceItem[];
};

async function parseError(response: Response, fallback: string) {
  try {
    const payload = await response.json();
    return payload.detail || fallback;
  } catch {
    return fallback;
  }
}

export async function login(username: string, password: string) {
  const body = new URLSearchParams();
  body.set("username", username);
  body.set("password", password);

  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body,
  });

  if (!response.ok) {
    throw new Error(await parseError(response, "Đăng nhập không thành công."));
  }

  return (await response.json()) as LoginResponse;
}

export async function register(mssv: string, email: string, password: string) {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ mssv, email, password }),
  });

  if (!response.ok) {
    throw new Error(await parseError(response, "Đăng kí không thành công."));
  }

  return (await response.json()) as RegisterResponse;
}

export async function sendQuestion(question: string, token: string) {
  const response = await fetch(
    `${API_BASE_URL}/chat?question=${encodeURIComponent(question)}`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    },
  );

  if (response.status === 401) {
    throw new Error("Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.");
  }

  if (!response.ok) {
    throw new Error(await parseError(response, "Không thể kết nối hệ thống tư vấn."));
  }

  return (await response.json()) as ChatResponse;
}

export async function getSessions(token: string) {
  const response = await fetch(`${API_BASE_URL}/history/sessions`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error("Không thể tải danh sách lịch sử chat.");
  return await response.json();
}

export async function getSessionMessages(sessionId: string, token: string) {
  const response = await fetch(`${API_BASE_URL}/history/messages/${sessionId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error("Không thể tải chi tiết đoạn chat.");
  return await response.json();
}

export type UploadTranscriptResponse = {
  status: string;
  message: string;
  data?: {
    gpa: string;
    total_passed: number;
  };
};

export async function uploadTranscript(file: File, token: string) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/grag/upload-transcript`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (response.status === 401) {
    throw new Error("Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.");
  }

  if (!response.ok) {
    throw new Error(await parseError(response, "Không thể tải lên bảng điểm."));
  }

  return (await response.json()) as UploadTranscriptResponse;
}

export async function sendQuestionStream(
  question: string,
  token: string,
  sessionId: string | null,
  onEvent: (event: string, data: any) => void
): Promise<void> {
  const url = `${API_BASE_URL}/chat/stream?question=${encodeURIComponent(question)}${sessionId ? `&session_id=${sessionId}` : ""}`;
  
  const response = await fetch(
    url,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  );

  if (response.status === 401) {
    throw new Error("Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.");
  }

  if (!response.ok) {
    throw new Error(await parseError(response, "Không thể kết nối hệ thống tư vấn."));
  }

  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const cleanLine = line.trim();
      if (cleanLine.startsWith("data: ")) {
        try {
          const payload = JSON.parse(cleanLine.slice(6));
          onEvent(payload.event, payload);
        } catch (e) {
          console.error("Lỗi parse stream:", e);
        }
      }
    }
  }
}


