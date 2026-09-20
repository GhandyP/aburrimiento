import type { ReactNode } from "react";

type BannerProps = {
  children: ReactNode;
  onDismiss?: () => void;
  tone?: "warning" | "error" | "info";
};

export function Banner({ children, onDismiss, tone = "warning" }: BannerProps) {
  return (
    <div className={`banner banner-${tone}`} role="alert">
      <span>{children}</span>
      {onDismiss && <button type="button" onClick={onDismiss} aria-label="Dismiss notification">×</button>}
    </div>
  );
}
