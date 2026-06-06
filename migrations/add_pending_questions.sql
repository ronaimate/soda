-- Migration: Add pending_questions column to ideas table
-- Run this against your PostgreSQL database

ALTER TABLE ideas ADD COLUMN IF NOT EXISTS pending_questions TEXT;

COMMENT ON COLUMN ideas.pending_questions IS 'JSON array of pending questions from architect AI';
