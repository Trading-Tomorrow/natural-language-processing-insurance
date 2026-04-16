export interface StatementCreate {
  role: string
  vehicle?: string | null
  text: string
}

export interface StatementRead {
  id: number
  role: string
  vehicle: string | null
  text: string
}

export interface ImageRead {
  id: number
  statement_id: number | null
  file_path: string
  damage_types: string[] | null
  yolo_raw_output: Record<string, unknown> | null
}

export interface CaseCreate {
  claim_id: string
  location: string
  incident_type: string
  detected_damages?: string[]
  statements?: StatementCreate[]
}

export interface CaseRead {
  id: number
  claim_id: string
  location: string
  incident_type: string
  detected_damages: string[]
  created_at: string
  probability_true: number | null
  verdict: string | null
  reasoning: string | null
  incongruences: string[] | null
  qwen_raw_output: string | null
  statements: StatementRead[]
  images: ImageRead[]
}

export interface AnalyzeResponse {
  case: CaseRead
  qwen_schema_valid: boolean
  qwen_validation_errors: string[]
  qwen_parse_error: string | null
}
