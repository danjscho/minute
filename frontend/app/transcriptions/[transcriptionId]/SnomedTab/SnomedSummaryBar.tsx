'use client'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { SnomedAnnotationResponse } from '@/lib/client'
import { Download } from 'lucide-react'

export function SnomedSummaryBar({
  annotations,
  transcriptionId,
}: {
  annotations: SnomedAnnotationResponse[]
  transcriptionId: string
}) {
  const entityCounts = annotations.reduce<Record<string, number>>((acc, a) => {
    const type = a.entity_type ?? 'unknown'
    acc[type] = (acc[type] ?? 0) + 1
    return acc
  }, {})

  const verifiedCount = annotations.filter((a) => a.is_verified).length

  const exportUrl = (format: string) =>
    `http://localhost:8080/transcriptions/${transcriptionId}/snomed-annotations/export?format=${format}`

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-gray-50 px-4 py-3">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm font-medium text-gray-700">
          {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
        </span>
        <span className="text-sm text-gray-500">{verifiedCount} verified</span>
        <div className="flex flex-wrap gap-1">
          {Object.entries(entityCounts)
            .sort(([, a], [, b]) => b - a)
            .map(([type, count]) => (
              <Badge key={type} variant="secondary" className="text-xs">
                {type.replace('_', ' ')} ({count})
              </Badge>
            ))}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" asChild>
          <a href={exportUrl('json')} download>
            <Download size={14} className="mr-1" />
            JSON
          </a>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <a href={exportUrl('csv')} download>
            <Download size={14} className="mr-1" />
            CSV
          </a>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <a href={exportUrl('fhir')} download>
            <Download size={14} className="mr-1" />
            FHIR
          </a>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <a href={exportUrl('html')} download>
            <Download size={14} className="mr-1" />
            HTML
          </a>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <a href={exportUrl('markdown')} download>
            <Download size={14} className="mr-1" />
            Markdown
          </a>
        </Button>
      </div>
    </div>
  )
}
