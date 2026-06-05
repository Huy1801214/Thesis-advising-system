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

export type ChatResponse = {
  session_id: string;
  status: string;
  answer?: string;
  data: unknown[];
  trace?: Array<{
    task_id: string;
    worker: string;
    sub_question: string;
    answer: string;
    status: string;
  }>;
  critic?: {
    passed: boolean;
    score: number;
    issues: Array<{
      type: string;
      severity: string;
      target: string;
      message: string;
      revision_instruction: string;
    }>;
    revised_answer?: string | null;
    error?: string | null;
  };
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
