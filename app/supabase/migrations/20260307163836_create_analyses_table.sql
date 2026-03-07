CREATE TABLE analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT now(),
    device_id TEXT,
    image_url TEXT NOT NULL,
    annotated_image_url TEXT,
    trichome_distribution JSONB,
    stigma_ratios JSONB,
    maturity_stage TEXT,
    recommendation TEXT,
    detections JSONB,
    corrections JSONB
);
