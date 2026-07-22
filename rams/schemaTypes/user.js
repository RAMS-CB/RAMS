import {defineField, defineType} from 'sanity'

export const user = defineType({
  name: 'user',
  title: 'User',
  type: 'document',
  fields: [
    defineField({
      name: 'username',
      title: 'Username',
      type: 'string',
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: 'email',
      title: 'Email',
      type: 'string',
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: 'full_name',
      title: 'Full Name',
      type: 'string',
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: 'hashed_password',
      title: 'Hashed Password',
      type: 'string',
    }),
    defineField({
      name: 'role',
      title: 'Role',
      type: 'string',
      options: {
        list: [
          {title: 'Super Admin', value: 'super_admin'},
          {title: 'Admin', value: 'admin'},
          {title: 'User', value: 'user'},
        ],
      },
      initialValue: 'user',
    }),
    defineField({
      name: 'hashed_refresh_token',
      title: 'Hashed Refresh Token',
      type: 'string',
    }),
    defineField({
      name: 'profession',
      title: 'Profession',
      type: 'string',
    }),
    defineField({
      name: 'level',
      title: 'Level',
      type: 'string',
    }),
    defineField({
      name: 'faculty_type',
      title: 'Faculty Type',
      type: 'string',
    }),
    defineField({
      name: 'age',
      title: 'Age',
      type: 'number',
    }),
    defineField({
      name: 'degree',
      title: 'Degree',
      type: 'string',
    }),
    defineField({
      name: 'source',
      title: 'Source',
      type: 'string',
    }),
    defineField({
      name: 'interested_programme',
      title: 'Interested Programme',
      type: 'string',
    }),
    defineField({
      name: 'created_at',
      title: 'Created At',
      type: 'datetime',
    }),
    defineField({
      name: 'last_active',
      title: 'Last Active',
      type: 'datetime',
    }),
    defineField({
      name: 'last_login',
      title: 'Last Login',
      type: 'datetime',
    }),
  ],
  preview: {
    select: {
      title: 'username',
      subtitle: 'email',
    },
  },
})
