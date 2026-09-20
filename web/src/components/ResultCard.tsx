import { LEVELS, type LevelId } from "../generated/schema";

type ResultCardProps = { levelId: LevelId };

export function ResultCard({ levelId }: ResultCardProps) {
  const level = LEVELS.find((candidate) => candidate.id === levelId);
  if (!level) return null;
  return (
    <section className="result-card" style={{ borderColor: level.color }} aria-labelledby="result-heading">
      <p className="eyebrow">Analysis result</p>
      <h2 id="result-heading">{level.labels.es} <span>({level.id})</span></h2>
      <p>{level.interpretation}</p>
      <p><strong>Recommended action:</strong> {level.recommendedAction}</p>
    </section>
  );
}
