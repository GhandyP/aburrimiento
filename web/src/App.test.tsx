import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import App from "./App";
import { GROUPS, INDICATOR_IDS, LEVELS } from "./generated/schema";

afterEach(() => cleanup());

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    if (String(input).endsWith("/health")) return new Response(JSON.stringify({ ok: true }), { status: 200 });
    return new Response(JSON.stringify({ nivel: "medio" }), { status: 200 });
  });
});

describe("analysis screen", () => {
  it("renders schema groups and every indicator", () => {
    render(<App />);
    for (const group of GROUPS) expect(screen.getByText(group.labels.es)).toBeInTheDocument();
    expect(screen.getAllByRole("slider")).toHaveLength(INDICATOR_IDS.length);
  });
  it("blocks an out-of-range value on the client", async () => {
    render(<App />);
    fireEvent.change(screen.getAllByRole("spinbutton")[0]!,  { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    expect(await screen.findByText(/between 0 and 1/)).toBeInTheDocument();
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });
  it("renders the level label and interpretation", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    await waitFor(() => expect(screen.getByRole("heading", { name: new RegExp(LEVELS[1]!.labels.es) })).toBeInTheDocument());
    expect(screen.getByText(LEVELS[1]!.interpretation)).toBeInTheDocument();
  });
  it("places server field errors beside the matching input", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/health")) return new Response("{}", { status: 200 });
      return new Response(JSON.stringify({ detail: [{ loc: ["body", "datos", INDICATOR_IDS[0]], msg: "Server field error" }] }), { status: 422 });
    });
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    expect(await screen.findByText("Server field error")).toBeInTheDocument();
    expect(screen.getAllByLabelText(/exact value/)[0]!).toHaveAttribute("aria-describedby", expect.stringContaining(INDICATOR_IDS[0]));
  });
});
