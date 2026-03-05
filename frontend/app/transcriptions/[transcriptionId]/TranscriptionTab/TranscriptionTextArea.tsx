import { DialogueEntryForm } from '@/app/transcriptions/[transcriptionId]/TranscriptionTab/TranscriptionTab'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import type { HighlightSegment } from '@/lib/snomed-highlight-utils'
import posthog from 'posthog-js'
import { useRef, useState } from 'react'
import { Control, Controller } from 'react-hook-form'

export const TranscriptionTextArea = ({
  index,
  control,
  highlightSegments,
}: {
  index: number
  control: Control<DialogueEntryForm>
  highlightSegments?: HighlightSegment[]
}) => {
  const [isEditing, setIsEditing] = useState(false)
  const pRef = useRef<HTMLParagraphElement>(null)

  const hasHighlights =
    !isEditing &&
    highlightSegments &&
    highlightSegments.some((s) => s.annotation)

  return (
    <div className="flex-1">
      <Controller
        render={({ field: { onChange, ...field } }) => (
          <p
            ref={pRef}
            className="flex-1 cursor-text rounded px-2 transition-all hover:bg-gray-100 hover:shadow-sm"
            suppressContentEditableWarning
            onClick={() => {
              const el = pRef.current
              if (!el || isEditing) return
              setIsEditing(true)
              el.setAttribute('contenteditable', 'true')
              el.innerText = field.value
              el.focus()
            }}
            onBlur={() => {
              const el = pRef.current
              if (!el) return
              el.setAttribute('contenteditable', 'false')
              setIsEditing(false)
              const newText = el.innerText.trim()

              if (newText !== field.value) {
                onChange(newText)

                posthog.capture('transcript_text_edited', {
                  entry_index: index,
                })
              }
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                e.currentTarget.blur()
              }
            }}
          >
            {hasHighlights
              ? highlightSegments.map((seg, i) =>
                  seg.annotation ? (
                    <Tooltip key={i}>
                      <TooltipTrigger asChild>
                        <span className="cursor-help rounded border-b-2 border-yellow-400 bg-yellow-100">
                          {seg.text}
                        </span>
                      </TooltipTrigger>
                      <TooltipContent>
                        <div className="text-xs">
                          <div className="font-semibold">
                            {seg.annotation.snomed_preferred_term}
                          </div>
                          <div className="text-gray-400">
                            {seg.annotation.snomed_concept_id}
                          </div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  ) : (
                    <span key={i}>{seg.text}</span>
                  )
                )
              : field.value}
          </p>
        )}
        control={control}
        name={`entries.${index}.text`}
      />
    </div>
  )
}
