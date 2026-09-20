import { describe, expect, it } from "vitest";
import { resolveApiBaseUrl } from "./config";

describe("resolveApiBaseUrl", () => {
  it("uses the desktop default when unset or whitespace", () => {
    expect(resolveApiBaseUrl()).toBe("http://127.0.0.1:8000");
    expect(resolveApiBaseUrl("  ")).toBe("http://127.0.0.1:8000");
  });
  it("strips trailing slashes", () => {
    expect(resolveApiBaseUrl("https://example.test///")).toBe("https://example.test");
  });
  it("falls back for relative or malformed URLs", () => {
    expect(resolveApiBaseUrl("/api")).toBe("http://127.0.0.1:8000");
    expect(resolveApiBaseUrl("not a url")).toBe("http://127.0.0.1:8000");
  });
});
