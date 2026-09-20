import { INDICATOR_IDS, SCHEMA_VERSION } from "./generated/schema";

function App() {
  return (
    <main className="app-shell">
      <p className="eyebrow">Aburrimiento</p>
      <h1>Group boredom analyzer</h1>
      <p className="status">
        Schema {SCHEMA_VERSION} · {INDICATOR_IDS.length} indicators ready
      </p>
    </main>
  );
}

export default App;
