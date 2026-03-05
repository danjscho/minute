import type { DialogueEntry, SnomedAnnotationResponse } from '@/lib/client'

export interface HighlightSegment {
  text: string
  annotation?: SnomedAnnotationResponse
}

/**
 * Compute the global character offset range for each dialogue entry.
 *
 * The SNOMED coding worker joins all dialogue entry texts with a single space:
 *   " ".join(entry["text"] for entry in dialogue_entries)
 *
 * So each entry's global start offset accounts for all preceding text plus
 * one space separator between each entry.
 */
export function computeEntryOffsets(
  entries: DialogueEntry[]
): { start: number; end: number }[] {
  let offset = 0
  return entries.map((entry, index) => {
    if (index > 0) {
      offset += 1 // space separator between entries
    }
    const start = offset
    const end = offset + entry.text.length
    offset = end
    return { start, end }
  })
}

/**
 * Given a dialogue entry's text and its global offset range,
 * find which SNOMED annotations overlap and split the text into
 * plain and highlighted segments.
 */
export function segmentEntryText(
  text: string,
  entryStart: number,
  entryEnd: number,
  annotations: SnomedAnnotationResponse[]
): HighlightSegment[] {
  const overlapping = annotations.filter(
    (a) =>
      a.start_char != null &&
      a.end_char != null &&
      a.start_char < entryEnd &&
      a.end_char > entryStart
  )

  if (overlapping.length === 0) {
    return [{ text }]
  }

  // Sort by start position
  const sorted = [...overlapping].sort(
    (a, b) => (a.start_char ?? 0) - (b.start_char ?? 0)
  )

  const segments: HighlightSegment[] = []
  let cursor = 0

  for (const ann of sorted) {
    const localStart = Math.max(0, (ann.start_char ?? 0) - entryStart)
    const localEnd = Math.min(text.length, (ann.end_char ?? 0) - entryStart)

    if (localStart > cursor) {
      segments.push({ text: text.slice(cursor, localStart) })
    }

    if (localEnd > localStart) {
      segments.push({
        text: text.slice(localStart, localEnd),
        annotation: ann,
      })
    }

    cursor = Math.max(cursor, localEnd)
  }

  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor) })
  }

  return segments
}
