import {defineField, defineType} from 'sanity'

export const docMetadata = defineType({
  name: 'doc_metadata',
  title: 'Document Metadata',
  type: 'document',
  fields: [
    defineField({
      name: 'doc_id',
      title: 'Document ID',
      type: 'string',
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: 'content_hash',
      title: 'Content Hash',
      type: 'string',
    }),
    defineField({
      name: 'last_updated_timestamp',
      title: 'Last Updated Timestamp',
      type: 'number',
    }),
  ],
  preview: {
    select: {
      title: 'doc_id',
      subtitle: 'content_hash',
    },
  },
})
