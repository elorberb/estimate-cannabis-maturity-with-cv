import { AnalyzeResponse } from "../api/types";

export class AnalysisResultStore {
  private static _latest: AnalyzeResponse | null = null;

  static set(result: AnalyzeResponse): void {
    AnalysisResultStore._latest = result;
  }

  static get(): AnalyzeResponse | null {
    return AnalysisResultStore._latest;
  }
}
