/** @type {import('next').NextConfig} */
const API_ORIGIN = process.env.BACKEND_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return [
      // REST: the browser always talks to the Next origin.
      { source: "/api/:path*", destination: `${API_ORIGIN}/api/:path*` },
      // Next rejects a ws:// rewrite destination outright ("Invalid rewrite"),
      // and it does not proxy protocol upgrades on every version. This entry
      // only covers plain HTTP requests to /ws; the client connects straight
      // to the FastAPI port (NEXT_PUBLIC_WS_URL, default :8000) for sockets.
      { source: "/ws", destination: `${API_ORIGIN}/ws` },
    ];
  },
};

export default nextConfig;
