'use client'

import { SnomedAnnotationCard } from '@/app/transcriptions/[transcriptionId]/SnomedTab/SnomedAnnotationCard'
import { SnomedSummaryBar } from '@/app/transcriptions/[transcriptionId]/SnomedTab/SnomedSummaryBar'
import { Button } from '@/components/ui/button'
import {
  getSnomedAnnotationsTranscriptionsTranscriptionIdSnomedAnnotationsGetOptions,
  triggerSnomedCodingTranscriptionsTranscriptionIdSnomedAnnotationsTriggerPostMutation,
  getSnomedAnnotationsTranscriptionsTranscriptionIdSnomedAnnotationsGetQueryKey,
} from '@/lib/client/@tanstack/react-query.gen'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FileText, LoaderCircle, Stethoscope } from 'lucide-react'

export function SnomedTab({ transcriptionId }: { transcriptionId: string }) {
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    ...getSnomedAnnotationsTranscriptionsTranscriptionIdSnomedAnnotationsGetOptions(
      { path: { transcription_id: transcriptionId } }
    ),
  })

  const invalidateAnnotations = () => {
    queryClient.invalidateQueries({
      queryKey:
        getSnomedAnnotationsTranscriptionsTranscriptionIdSnomedAnnotationsGetQueryKey(
          { path: { transcription_id: transcriptionId } }
        ),
    })
  }

  const { mutate: triggerTranscriptCoding, isPending: isTranscriptCoding } =
    useMutation({
      ...triggerSnomedCodingTranscriptionsTranscriptionIdSnomedAnnotationsTriggerPostMutation(),
      onSuccess: invalidateAnnotations,
    })

  const { mutate: triggerMinuteCoding, isPending: isMinuteCoding } =
    useMutation({
      ...triggerSnomedCodingTranscriptionsTranscriptionIdSnomedAnnotationsTriggerPostMutation(),
      onSuccess: invalidateAnnotations,
    })

  if (isLoading) {
    return (
      <div className="flex h-48 items-center justify-center">
        <LoaderCircle size={40} className="animate-spin text-gray-400" />
      </div>
    )
  }

  const annotations = data?.annotations ?? []

  if (annotations.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-16">
        <Stethoscope size={48} className="text-gray-300" />
        <p className="text-gray-500">
          No clinical codes found for this transcription.
        </p>
        <div className="flex gap-3">
          <Button
            onClick={() =>
              triggerTranscriptCoding({
                path: { transcription_id: transcriptionId },
                query: { source_type: 'transcript' },
              })
            }
            disabled={isTranscriptCoding}
            variant="outline"
          >
            {isTranscriptCoding ? (
              <LoaderCircle size={16} className="mr-2 animate-spin" />
            ) : (
              <FileText size={16} className="mr-2" />
            )}
            {isTranscriptCoding ? 'Coding...' : 'Code Transcript'}
          </Button>
          <Button
            onClick={() =>
              triggerMinuteCoding({
                path: { transcription_id: transcriptionId },
                query: { source_type: 'minute' },
              })
            }
            disabled={isMinuteCoding}
          >
            {isMinuteCoding ? (
              <LoaderCircle size={16} className="mr-2 animate-spin" />
            ) : (
              <Stethoscope size={16} className="mr-2" />
            )}
            {isMinuteCoding ? 'Coding...' : 'Code Minutes'}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4 py-4">
      <SnomedSummaryBar
        annotations={annotations}
        transcriptionId={transcriptionId}
      />
      <div className="flex gap-2">
        <Button
          onClick={() =>
            triggerTranscriptCoding({
              path: { transcription_id: transcriptionId },
              query: { source_type: 'transcript' },
            })
          }
          disabled={isTranscriptCoding}
          variant="outline"
          size="sm"
        >
          {isTranscriptCoding ? (
            <LoaderCircle size={14} className="mr-1 animate-spin" />
          ) : (
            <FileText size={14} className="mr-1" />
          )}
          {isTranscriptCoding ? 'Coding...' : 'Code Transcript'}
        </Button>
        <Button
          onClick={() =>
            triggerMinuteCoding({
              path: { transcription_id: transcriptionId },
              query: { source_type: 'minute' },
            })
          }
          disabled={isMinuteCoding}
          variant="outline"
          size="sm"
        >
          {isMinuteCoding ? (
            <LoaderCircle size={14} className="mr-1 animate-spin" />
          ) : (
            <Stethoscope size={14} className="mr-1" />
          )}
          {isMinuteCoding ? 'Coding...' : 'Code Minutes'}
        </Button>
      </div>
      <div className="space-y-3">
        {annotations.map((annotation) => (
          <SnomedAnnotationCard
            key={annotation.id}
            annotation={annotation}
            transcriptionId={transcriptionId}
          />
        ))}
      </div>
    </div>
  )
}
