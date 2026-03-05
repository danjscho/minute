'use client'

import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { useDebouncedValue } from '@/hooks/use-debounced-value'
import type { SnomedConceptSearchResult } from '@/lib/client'
import { searchSnomedConceptsSnomedConceptsSearchGetOptions } from '@/lib/client/@tanstack/react-query.gen'
import { useQuery } from '@tanstack/react-query'
import { LoaderCircle, Search } from 'lucide-react'
import { useCallback, useState } from 'react'

interface SnomedConceptSearchProps {
  onSelect: (concept: SnomedConceptSearchResult) => void
  placeholder?: string
}

export function SnomedConceptSearch({
  onSelect,
  placeholder = 'Search SNOMED concepts...',
}: SnomedConceptSearchProps) {
  const [query, setQuery] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const debouncedQuery = useDebouncedValue(query, 300)

  const { data, isLoading } = useQuery({
    ...searchSnomedConceptsSnomedConceptsSearchGetOptions({
      query: { q: debouncedQuery, limit: 10 },
    }),
    enabled: debouncedQuery.length >= 2,
  })

  const handleSelect = useCallback(
    (concept: SnomedConceptSearchResult) => {
      onSelect(concept)
      setQuery('')
      setIsOpen(false)
    },
    [onSelect]
  )

  return (
    <Popover
      open={isOpen && debouncedQuery.length >= 2}
      onOpenChange={setIsOpen}
    >
      <PopoverAnchor asChild>
        <div className="relative">
          <Search
            size={16}
            className="absolute top-1/2 left-3 -translate-y-1/2 text-gray-400"
          />
          <Input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setIsOpen(true)
            }}
            placeholder={placeholder}
            className="pl-9"
            onFocus={() => {
              if (query.length >= 2) setIsOpen(true)
            }}
          />
          {isLoading && (
            <LoaderCircle
              size={16}
              className="absolute top-1/2 right-3 -translate-y-1/2 animate-spin text-gray-400"
            />
          )}
        </div>
      </PopoverAnchor>
      <PopoverContent
        className="w-[var(--radix-popover-trigger-width)] p-0"
        align="start"
        sideOffset={4}
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        {data?.results?.length ? (
          <ul className="max-h-64 overflow-y-auto py-1">
            {data.results.map((concept) => (
              <li key={concept.concept_id}>
                <button
                  className="flex w-full items-start gap-2 px-3 py-2 text-left text-sm hover:bg-gray-100"
                  onClick={() => handleSelect(concept)}
                >
                  <div className="min-w-0 flex-1">
                    <div className="font-medium">{concept.preferred_term}</div>
                    <div className="font-mono text-xs text-gray-500">
                      {concept.concept_id}
                    </div>
                  </div>
                  <Badge variant="secondary" className="shrink-0 text-xs">
                    {concept.semantic_tag}
                  </Badge>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <div className="px-3 py-4 text-center text-sm text-gray-500">
            No concepts found
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}
