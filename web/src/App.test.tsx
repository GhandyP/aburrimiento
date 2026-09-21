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
    if (String(input).endsWith("/samples")) return new Response(JSON.stringify({ id: 7, nivel: "medio" }), { status: 200 });
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
  it("does not show the save control before analysis and offers it afterward", async () => {
    render(<App />);
    expect(screen.queryByRole("button", { name: "Save observation" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    expect(await screen.findByRole("button", { name: "Save observation" })).toBeInTheDocument();
  });
  it("saves the selected observed level or omits it when declined", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    await screen.findByRole("button", { name: "Save observation" });
    fireEvent.change(screen.getByLabelText("What was the group actually like?"), { target: { value: LEVELS[2]!.id } });
    fireEvent.click(screen.getByRole("button", { name: "Save observation" }));
    await screen.findByText(/id 7/);
    const savedBody = JSON.parse(String(fetchMock.mock.calls.find(([input]) => String(input).endsWith("/samples"))?.[1]?.body));
    expect(savedBody.nivel_observado).toBe(LEVELS[2]!.id);
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    await screen.findByRole("button", { name: "Save observation" });
    fireEvent.click(screen.getByRole("button", { name: "Save observation" }));
    await screen.findByText(/id 7/);
    const sampleBodies = fetchMock.mock.calls.filter(([input]) => String(input).endsWith("/samples")).map(([, init]) => JSON.parse(String(init?.body)));
    expect(sampleBodies.at(-1)).not.toHaveProperty("nivel_observado");
  });
  it("shows save failures and clears an old confirmation on a new analysis", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockImplementation(async (input) => {
      if (String(input).endsWith("/health")) return new Response("{}", { status: 200 });
      if (String(input).endsWith("/samples")) return new Response(JSON.stringify({ detail: "Save failed" }), { status: 500 });
      return new Response(JSON.stringify({ nivel: "medio" }), { status: 200 });
    });
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    await screen.findByRole("button", { name: "Save observation" });
    fireEvent.click(screen.getByRole("button", { name: "Save observation" }));
    expect(await screen.findByText(/Save failed/)).toBeInTheDocument();
    fetchMock.mockImplementation(async (input) => {
      if (String(input).endsWith("/health")) return new Response("{}", { status: 200 });
      return new Response(JSON.stringify({ nivel: "medio" }), { status: 200 });
    });
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    await waitFor(() => expect(screen.queryByText(/Save failed/)).not.toBeInTheDocument());
    expect(screen.queryByText(/id 7/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save observation" })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalled();
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
