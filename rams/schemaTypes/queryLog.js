import {defineField, defineType} from 'sanity'

export const queryLog = defineType({
  name: 'query_log',
  title: 'Query Log',
  type: 'document',
  fields: [
    defineField({
      name: 'user_id',
      title: 'User ID',
      type: 'string',
    }),
    defineField({
      name: 'query',
      title: 'Query',
      type: 'text',
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: 'timestamp',
      title: 'Timestamp',
      type: 'datetime',
    }),
  ],
  preview: {
    select: {
      title: 'query',
      subtitle: 'timestamp',
    },
  },
})
