import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(v: number, decimals = 1): string {
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(decimals)}%`;
}

export function colorValue(v: number): string {
  if (v > 0) return "text-green-400";
  if (v < 0) return "text-red-400";
  return "text-gray-400";
}

export function formatNumber(v: number, decimals = 0): string {
  return v.toLocaleString("ko-KR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}
