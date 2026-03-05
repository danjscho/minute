'use client'

import { SnomedConceptSearch } from '@/app/transcriptions/[transcriptionId]/SnomedTab/SnomedConceptSearch'
import { SnomedVerifyButton } from '@/app/transcriptions/[transcriptionId]/SnomedTab/SnomedVerifyButton'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import type {
  AlternativeSnomedConcept,
  SnomedAnnotationResponse,
  SnomedConceptSearchResult,
} from '@/lib/client'
import {
  getSnomedAnnotationsTranscriptionsTranscriptionIdSnomedAnnotationsGetQueryKey,
  verifyAnnotationSnomedAnnotationsAnnotationIdVerifyPatchMutation,
} from '@/lib/client/@tanstack/react-query.gen'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ChevronDown, ExternalLink, Search } from 'lucide-react'
import { useState } from 'react'

const ENTITY_TYPE_STYLES: Record<string, string> = {
  finding: 'bg-blue-100 text-blue-800 hover:bg-blue-100 border-blue-200',
  procedure: 'bg-green-100 text-green-800 hover:bg-green-100 border-green-200',
  body_structure:
    'bg-orange-100 text-orange-800 hover:bg-orange-100 border-orange-200',
  disorder: 'bg-red-100 text-red-800 hover:bg-red-100 border-red-200',
  substance:
    'bg-purple-100 text-purple-800 hover:bg-purple-100 border-purple-200',
  observable_entity:
    'bg-teal-100 text-teal-800 hover:bg-teal-100 border-teal-200',
}

function getEntityBadgeClass(entityType: string | null): string {
  if (!entityType) return ''
  return (
    ENTITY_TYPE_STYLES[entityType] ??
    'bg-gray-100 text-gray-800 hover:bg-gray-100 border-gray-200'
  )
}

function snomedBrowserUrl(conceptId: string): string {
  return `https://browser.ihtsdotools.org/?perspective=full&conceptId1=${conceptId}`
}

export function SnomedAnnotationCard({
  annotation,
  transcriptionId,
}: {
  annotation: SnomedAnnotationResponse
  transcriptionId: string
}) {
  const [isOpen, setIsOpen] = useState(false)
  const [showSearch, setShowSearch] = useState(false)
  const alternatives = annotation.alternative_concepts ?? []
  const queryClient = useQueryClient()

  const { mutate: overrideConcept } = useMutation({
    ...verifyAnnotationSnomedAnnotationsAnnotationIdVerifyPatchMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey:
          getSnomedAnnotationsTranscriptionsTranscriptionIdSnomedAnnotationsGetQueryKey(
            { path: { transcription_id: transcriptionId } }
          ),
      })
      setShowSearch(false)
    },
  })

  const handleConceptSelect = (concept: SnomedConceptSearchResult) => {
    overrideConcept({
      path: { annotation_id: annotation.id },
      body: {
        is_verified: true,
        snomed_concept_id: concept.concept_id,
        snomed_preferred_term: concept.preferred_term,
        snomed_fsn: concept.fsn,
      },
    })
  }

  return (
    <div className="rounded-lg border bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex items-center gap-2">
            <span className="font-semibold text-gray-900">
              &ldquo;{annotation.text_span}&rdquo;
            </span>
            {annotation.entity_type && (
              <Badge
                variant="outline"
                className={getEntityBadgeClass(annotation.entity_type)}
              >
                {annotation.entity_type.replace('_', ' ')}
              </Badge>
            )}
            {annotation.source_type && (
              <Badge
                variant="outline"
                className={
                  annotation.source_type === 'minute'
                    ? 'border-violet-200 bg-violet-50 text-violet-700'
                    : 'border-gray-200 bg-gray-50 text-gray-600'
                }
              >
                {annotation.source_type === 'minute' ? 'Minutes' : 'Transcript'}
              </Badge>
            )}
            {annotation.is_verified && (
              <Badge
                variant="outline"
                className="border-green-300 bg-green-50 text-green-700"
              >
                Verified
              </Badge>
            )}
          </div>

          {annotation.snomed_concept_id && (
            <div className="mb-1 flex items-center gap-2 text-sm">
              <a
                href={snomedBrowserUrl(annotation.snomed_concept_id)}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 font-mono text-blue-600 hover:underline"
              >
                {annotation.snomed_concept_id}
                <ExternalLink size={12} />
              </a>
              <span className="text-gray-600">
                {annotation.snomed_preferred_term}
              </span>
            </div>
          )}

          {annotation.snomed_fsn && (
            <p className="text-xs text-gray-500">{annotation.snomed_fsn}</p>
          )}

          {annotation.confidence_score != null && (
            <div className="mt-2 flex items-center gap-2">
              <div className="h-1.5 w-24 rounded-full bg-gray-200">
                <div
                  className="h-1.5 rounded-full bg-blue-500"
                  style={{
                    width: `${Math.round(annotation.confidence_score * 100)}%`,
                  }}
                />
              </div>
              <span className="text-xs text-gray-500">
                {Math.round(annotation.confidence_score * 100)}%
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="text-gray-500 hover:text-gray-700"
            onClick={() => setShowSearch(!showSearch)}
          >
            <Search size={14} />
          </Button>
          <SnomedVerifyButton
            annotation={annotation}
            transcriptionId={transcriptionId}
          />
        </div>
      </div>

      {showSearch && (
        <div className="mt-3">
          <SnomedConceptSearch
            onSelect={handleConceptSelect}
            placeholder="Search for a different concept..."
          />
        </div>
      )}

      {alternatives.length > 0 && (
        <Collapsible open={isOpen} onOpenChange={setIsOpen} className="mt-3">
          <CollapsibleTrigger className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700">
            <ChevronDown
              size={14}
              className={`transition-transform ${isOpen ? 'rotate-180' : ''}`}
            />
            {alternatives.length} alternative
            {alternatives.length !== 1 ? 's' : ''}
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-2">
            <AlternativesList
              alternatives={alternatives}
              annotation={annotation}
              transcriptionId={transcriptionId}
            />
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  )
}

function AlternativesList({
  alternatives,
  annotation,
  transcriptionId,
}: {
  alternatives: AlternativeSnomedConcept[]
  annotation: SnomedAnnotationResponse
  transcriptionId: string
}) {
  return (
    <div className="space-y-1 rounded border bg-gray-50 p-2">
      {alternatives.map((alt) => (
        <div
          key={alt.concept_id}
          className="flex items-center justify-between gap-2 text-sm"
        >
          <div className="flex items-center gap-2">
            <a
              href={snomedBrowserUrl(alt.concept_id)}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 font-mono text-xs text-blue-600 hover:underline"
            >
              {alt.concept_id}
              <ExternalLink size={10} />
            </a>
            <span className="text-gray-600">{alt.preferred_term}</span>
            <span className="text-xs text-gray-400">
              {Math.round(alt.confidence_score * 100)}%
            </span>
          </div>
          <SnomedVerifyButton
            annotation={annotation}
            transcriptionId={transcriptionId}
            overrideConcept={alt}
            size="sm"
          />
        </div>
      ))}
    </div>
  )
}
