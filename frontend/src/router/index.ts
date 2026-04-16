import { createRouter, createWebHistory } from 'vue-router'
import CaseDetailView from '@/views/CaseDetailView.vue'
import NewCaseView from '@/views/NewCaseView.vue'
import EmptyView from '@/views/EmptyView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: EmptyView },
    { path: '/cases/new', component: NewCaseView },
    { path: '/cases/:id', component: CaseDetailView, props: true },
  ],
})
