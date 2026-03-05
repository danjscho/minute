'use client'

import { Button } from '@/components/ui/button'
import type {
  AlternativeSnomedConcept,
  SnomedAnnotationResponse,
} from '@/lib/client'
import {
  deleteAnnotationSnomedAnnotationsAnnotationIdDeleteMutation,
  getSnomedAnnotationsTranscriptionsTranscriptionIdSnomedAnnotationsGetQueryKey,
  verifyAnnotationSnomedAnnotationsAnnotationIdVerifyPatchMutation,
} from '@/lib/client/@tanstack/react-query.gen'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Check, RotateCcw, X } from 'lucide-react'

export function SnomedVerifyButton({
  annotation,
  transcriptionId,
  overrideConcept,
  size = 'default',
}: {
  annotation: SnomedAnnotationResponse
  transcriptionId: string
  overrideConcept?: AlternativeSnomedConcept
  size?: 'default' | 'sm'
}) {
  const queryClient = useQueryClient()

  const invalidateAnnotations = () => {
    queryClient.invalidateQueries({
      queryKey:
        getSnomedAnnotationsTranscriptionsTranscriptionIdSnomedAnnotationsGetQueryKey(
          { path: { transcription_id: transcriptionId } }
        ),
    })
  }

  const { mutate: verify, isPending: isVerifying } = useMutation({
    ...verifyAnnotationSnomedAnnotationsAnnotationIdVerifyPatchMutation(),
    onSuccess: invalidateAnnotations,
  })

  const { mutate: deleteAnnotation, isPending: isDeleting } = useMutation({
    ...deleteAnnotationSnomedAnnotationsAnnotationIdDeleteMutation(),
    onSuccess: invalidateAnnotations,
  })

  const isPending = isVerifying || isDeleting
  const iconSize = size === 'sm' ? 12 : 14
  const buttonSize = size === 'sm' ? 'sm' : 'default'

  // If this is for an alternative concept, show "Use this" button
  if (overrideConcept) {
    return (
      <Button
        variant="ghost"
        size="sm"
        disabled={isPending}
        className="h-6 px-2 text-xs"
        onClick={() =>
          verify({
            path: { annotation_id: annotation.id },
            body: {
              is_verified: true,
              snomed_concept_id: overrideConcept.concept_id,
              snomed_preferred_term: overrideConcept.preferred_term,
              snomed_fsn: overrideConcept.fsn,
            },
          })
        }
      >
        <Check size={iconSize} className="mr-1" />
        Use
      </Button>
    )
  }

  // Main verify/reject buttons
  return (
    <div className="flex items-center gap-1">
      {annotation.is_verified ? (
        <Button
          variant="ghost"
          size={buttonSize}
          disabled={isPending}
          className="text-gray-500 hover:text-gray-700"
          onClick={() =>
            verify({
              path: { annotation_id: annotation.id },
              body: { is_verified: false },
            })
          }
        >
          <RotateCcw size={iconSize} className="mr-1" />
          Undo
        </Button>
      ) : (
        <>
          <Button
            variant="ghost"
            size={buttonSize}
            disabled={isPending}
            className="text-green-600 hover:bg-green-50 hover:text-green-700"
            onClick={() =>
              verify({
                path: { annotation_id: annotation.id },
                body: { is_verified: true },
              })
            }
          >
            <Check size={iconSize} />
          </Button>
          <Button
            variant="ghost"
            size={buttonSize}
            disabled={isPending}
            className="text-red-500 hover:bg-red-50 hover:text-red-700"
            onClick={() =>
              deleteAnnotation({
                path: { annotation_id: annotation.id },
              })
            }
          >
            <X size={iconSize} />
          </Button>
        </>
      )}
    </div>
  )
}
