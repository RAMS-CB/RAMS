import {defineField, defineType} from 'sanity'

export const documentSource = defineType({
  name: 'document_source',
  title: 'Document Source',
  type: 'document',
  fields: [
    defineField({
      name: 'doc_id',
      title: 'Document ID',
      type: 'string',
      description: 'Unique string identifier (e.g. google_doc_main)',
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: 'title',
      title: 'Title',
      type: 'string',
    }),
    defineField({
      name: 'url',
      title: 'URL',
      type: 'string',
    }),
  ],
  preview: {
    select: {
      title: 'title',
      subtitle: 'doc_id',
    },
  },
})
