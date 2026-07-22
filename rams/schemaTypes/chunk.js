import {defineField, defineType} from 'sanity'

export const chunk = defineType({
  name: 'chunk',
  title: 'Chunk',
  type: 'document',
  fields: [
    defineField({
      name: 'doc_id',
      title: 'Document ID',
      type: 'string',
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: 'chunk_index',
      title: 'Chunk Index',
      type: 'number',
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: 'text_content',
      title: 'Text Content',
      type: 'text',
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: 'embedding',
      title: 'Vector Embedding',
      type: 'array',
      of: [{type: 'number'}],
      description: 'Gemini vector embedding representation',
    }),
    defineField({
      name: 'metadata',
      title: 'Metadata',
      type: 'object',
      fields: [
        defineField({
          name: 'content_hash',
          title: 'Content Hash',
          type: 'string',
        }),
        defineField({
          name: 'title',
          title: 'Title',
          type: 'string',
        }),
        defineField({
          name: 'source_title',
          title: 'Source Title',
          type: 'string',
        }),
        defineField({
          name: 'url',
          title: 'URL',
          type: 'string',
        }),
        defineField({
          name: 'section',
          title: 'Section',
          type: 'string',
        }),
      ],
    }),
    defineField({
      name: 'created_at',
      title: 'Created At',
      type: 'datetime',
    }),
  ],
  preview: {
    select: {
      title: 'doc_id',
      subtitle: 'chunk_index',
    },
    prepare({title, subtitle}) {
      return {
        title: `${title || 'Unknown Doc'} (Chunk #${subtitle ?? 'N/A'})`,
      }
    },
  },
})
